#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
int main(){
    int dev=0; hipSetDevice(dev);
    ncclUniqueId id; ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t rc = ncclCommInitRank(&comm, 1, id, 0);   // world=1, valid on 1 GPU
    if(rc!=ncclSuccess){ printf("INIT_FAIL: %s\n", ncclGetErrorString(rc)); return 1; }
    printf("INIT_OK\n");
    float *d; hipMalloc(&d, 16*sizeof(float));
    float h[16]; for(int i=0;i<16;i++) h[i]=2.0f;
    hipMemcpy(d,h,sizeof(h),hipMemcpyHostToDevice);
    hipStream_t s; hipStreamCreate(&s);
    rc = ncclAllReduce(d,d,16,ncclFloat,ncclSum,comm,s);
    hipStreamSynchronize(s);
    if(rc!=ncclSuccess){ printf("ALLREDUCE_FAIL: %s\n", ncclGetErrorString(rc)); return 1; }
    hipMemcpy(h,d,sizeof(h),hipMemcpyDeviceToHost);
    printf("ALLREDUCE_OK: out[0]=%.1f (expect 2.0 for world=1)\n", h[0]);
    ncclCommDestroy(comm);
    return 0;
}
