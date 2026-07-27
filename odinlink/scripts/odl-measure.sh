#!/usr/bin/env bash
# Latency + bandwidth over the OdinLink link.
#   node A:  ./odl-measure.sh server
#   node B:  ./odl-measure.sh client
# Device indices are assigned by probe order and DIFFER between nodes, so we
# read them rather than assuming 0.
set -u
CLI="${CLI:-./build/cli/odl_tb5_cli}"
DEV=$(ls /dev/odl_tb5_* 2>/dev/null | head -1 | grep -oE '[0-9]+$')
[ -n "${DEV:-}" ] || { echo "no /dev/odl_tb5_* - link not up"; exit 1; }

case "${1:-}" in
  server) exec "$CLI" server -d "$DEV" -v ;;
  client)
    echo "=== latency (1000 iterations) ==="
    "$CLI" client -d "$DEV" -t latency -i 1000
    echo "=== bandwidth ==="
    "$CLI" client -d "$DEV" -t bandwidth -b 4K,64K,1M,4M -D 5
    ;;
  *) echo "usage: $0 server|client"; exit 2 ;;
esac
