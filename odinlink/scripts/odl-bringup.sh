#!/usr/bin/env bash
# Bind exactly one odl_tb5 service and wait for the link to reach READY.
# With two Thunderbolt cables both services report route=2 and the login
# handshake never completes (FINDINGS.md BUG 2) - so we unbind the extras.
set -u
D=/sys/bus/thunderbolt/drivers/odl_tb5

[ -d "$D" ] || { echo "odl_tb5 not loaded - run odl-reload.sh first"; exit 1; }

bound=$(ls "$D" | grep -cE '^[01]-')
if [ "$bound" -gt 1 ]; then
  echo "unbinding extra services (keeping domain 0)"
  for s in $(ls "$D" | grep -E '^1-'); do
    echo "$s" | sudo tee "$D/unbind" >/dev/null 2>&1
  done
  sleep 2
fi

echo "bound: $(ls $D | grep -E '^[01]-' | tr '\n' ' ')"
echo "dev:   $(ls /dev/odl_tb5_* 2>/dev/null | tr '\n' ' ')"

echo "waiting for READY (needs the peer to be up too)..."
for i in $(seq 1 20); do
  if sudo dmesg | tail -50 | grep -q "entering READY state"; then
    echo "READY after ~$((i*5))s"; sudo dmesg | grep OdinLink | tail -4; exit 0
  fi
  sleep 5
done
echo "NOT READY - check: sudo dmesg | grep -E 'odl_tb5|OdinLink' | tail -20"
echo "  'enable_paths failed (-12)'  -> hop-ID leak, run odl-reload.sh"
echo "  'login request failed: -110' -> route collision, unbind a service"
exit 1
