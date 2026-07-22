// End-to-end test for GGML_OP_ALLREDUCE over the cross-process world communicator.
//
// Proves the core mechanism for cross-NODE tensor parallelism in llama.cpp:
// the head process runs its graph on the local CUDA backend (world rank 0)
// while shipping an identical graph to a remote ggml-rpc-server (world rank 1)
// in ONE GRAPH_COMPUTE round trip; the two ranks rendezvous *inside* the
// embedded allreduce nodes via NCCL/RCCL — no per-layer RPC round trips.
//
// Usage:
//   peer:  GGML_NCCL_RANK=1 GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=<head-ip>:29500 \
//          ggml-rpc-server -H <peer-ip> -p 50052
//   head:  GGML_NCCL_RANK=0 GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=<head-ip>:29500 \
//          test-world-allreduce --rpc <peer-ip>:50052 [--layers 60] [--iters 20] [--ne 6144]
//
// Local smoke mode (no env, no rpc): test-world-allreduce   -> builds the graph on
// the CUDA backend and expects allreduce to fail gracefully (unsupported), exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct graph_bundle {
    ggml_context *        ctx     = nullptr;
    ggml_cgraph  *        graph   = nullptr;
    ggml_tensor  *        input   = nullptr;
    ggml_tensor  *        output  = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
};

// input -> [ allreduce -> scale(0.5) ] x layers.
// With rank0=1.0 and rank1=2.0 the value is 3.0 after each allreduce and 1.5
// after each scale -> steady state; final output must be 1.5 on both ranks.
static graph_bundle build_graph(ggml_backend_t backend, int64_t ne, int layers) {
    graph_bundle b;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(layers * 2 + 8) + ggml_graph_overhead_custom(layers * 2 + 8, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    b.ctx = ggml_init(ip);

    b.input = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, ne);
    ggml_set_name(b.input, "in");

    ggml_tensor * cur = b.input;
    for (int l = 0; l < layers; ++l) {
        cur = ggml_allreduce(b.ctx, cur);
        cur = ggml_scale(b.ctx, cur, 0.5f);
    }
    b.output = cur;
    ggml_set_name(b.output, "out");

    b.graph = ggml_new_graph_custom(b.ctx, layers * 2 + 8, false);
    ggml_build_forward_expand(b.graph, b.output);

    b.buffer = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (b.buffer == nullptr) {
        fprintf(stderr, "FATAL: failed to allocate tensors on backend %s\n", ggml_backend_name(backend));
        exit(1);
    }
    return b;
}

static void fill_input(graph_bundle & b, float v, int64_t ne) {
    std::vector<float> host(ne, v);
    ggml_backend_tensor_set(b.input, host.data(), 0, ne * sizeof(float));
}

int main(int argc, char ** argv) {
    const char * rpc_endpoint = nullptr;
    int     layers = 60;
    int     iters  = 20;
    int64_t ne     = 6144;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rpc")    && i + 1 < argc) { rpc_endpoint = argv[++i]; }
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { layers = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--iters")  && i + 1 < argc) { iters  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ne")     && i + 1 < argc) { ne     = atoll(argv[++i]); }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }

    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cuda == nullptr) {
        fprintf(stderr, "FATAL: no CUDA/HIP backend\n");
        return 1;
    }

    const bool world = getenv("GGML_NCCL_RANK") != nullptr;
    if (!world) {
        printf("no GGML_NCCL_RANK set: local smoke mode (graph build only)\n");
        graph_bundle b = build_graph(cuda, ne, 1);
        printf("graph with ALLREDUCE node built + allocated OK (op registered)\n");
        ggml_backend_buffer_free(b.buffer);
        ggml_free(b.ctx);
        ggml_backend_free(cuda);
        return 0;
    }

    // ---- each rank runs its own local graph; they rendezvous in the allreduce ----
    // Distinct per-rank inputs so the correctness check is meaningful:
    // rank r contributes (r+1). Sum over 2 ranks = 1+2 = 3, then *0.5 -> 1.5 steady state.
    const int this_rank = atoi(getenv("GGML_NCCL_RANK"));
    graph_bundle local = build_graph(cuda, ne, layers);
    fill_input(local, (float)(this_rank + 1), ne);

    ggml_backend_t rpc = nullptr;
    graph_bundle remote;
    if (rpc_endpoint != nullptr) {
        rpc = ggml_backend_rpc_init(rpc_endpoint, 0);
        if (rpc == nullptr) {
            fprintf(stderr, "FATAL: cannot connect rpc %s\n", rpc_endpoint);
            return 1;
        }
        remote = build_graph(rpc, ne, layers);
        fill_input(remote, 2.0f, ne);
        printf("remote graph allocated on %s (%d allreduce nodes)\n", rpc_endpoint, layers);
    }

    printf("world run: layers=%d iters=%d ne=%lld (%.1f KB/allreduce)\n",
           layers, iters, (long long) ne, ne * 4.0 / 1024.0);

    // warmup + correctness (1 iteration)
    {
        std::thread rt;
        if (rpc) {
            rt = std::thread([&] { ggml_backend_graph_compute(rpc, remote.graph); });
        }
        ggml_status st = ggml_backend_graph_compute(cuda, local.graph);
        if (rt.joinable()) rt.join();
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: local graph_compute failed\n");
            return 1;
        }
        std::vector<float> out(ne);
        ggml_backend_tensor_get(local.output, out.data(), 0, ne * sizeof(float));
        const float expect = 1.5f;   // (1+2) * 0.5, steady state
        if (fabsf(out[0] - expect) > 1e-3f || fabsf(out[ne-1] - expect) > 1e-3f) {
            fprintf(stderr, "FAIL: local out=%f/%f expect=%f (allreduce broken)\n", out[0], out[ne-1], expect);
            return 1;
        }
        if (rpc) {
            std::vector<float> rout(ne);
            ggml_backend_tensor_get(remote.output, rout.data(), 0, ne * sizeof(float));
            if (fabsf(rout[0] - expect) > 1e-3f) {
                fprintf(stderr, "FAIL: remote out=%f expect=%f\n", rout[0], expect);
                return 1;
            }
        }
        printf("CORRECTNESS OK: both ranks reduced to %.2f\n", 1.5f);
    }

    // timing
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::thread rt;
        if (rpc) {
            rt = std::thread([&] { ggml_backend_graph_compute(rpc, remote.graph); });
        }
        ggml_backend_graph_compute(cuda, local.graph);
        if (rt.joinable()) rt.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double per_graph = sec / iters * 1000.0;
    const double per_ar    = sec / iters / layers * 1e6;

    printf("RESULT: %.2f ms/graph (%d allreduces)  ->  %.0f us/allreduce  [incl. 1 RPC round trip/graph]\n",
           per_graph, layers, per_ar);

    if (rpc) ggml_backend_free(rpc);
    ggml_backend_free(cuda);
    return 0;
}
