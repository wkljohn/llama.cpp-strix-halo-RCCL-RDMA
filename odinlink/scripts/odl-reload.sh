#!/usr/bin/env bash
# No-reboot recovery for the XDomain hop-ID leak (FINDINGS.md BUG 1).
# The driver leaks hop-IDs on unload/re-probe, so the 2nd insmod after a boot
# fails with `enable_paths failed (-12)` (ENOMEM). Reloading the whole
# Thunderbolt stack tears down the domain and frees them - no reboot required.
#
# WARNING: this also resets thunderbolt_net, i.e. your IP bond. Keep a
# non-Thunderbolt path (LAN/Wi-Fi) to the peer before running it.
set -u
KO="${1:-driver/odl_tb5.ko}"

echo "[1/4] unloading Thunderbolt stack"
sudo rmmod odl_tb5        2>/dev/null
sudo rmmod thunderbolt_net 2>/dev/null
sudo rmmod thunderbolt     2>/dev/null
sleep 2

echo "[2/4] reloading"
sudo modprobe thunderbolt      ; sleep 4
sudo modprobe thunderbolt_net  ; sleep 4

echo "[3/4] loading odl_tb5 (e2e=0)"
sudo insmod "$KO" e2e=0 || { echo "insmod failed"; exit 1; }
sleep 3

echo "[4/4] bond sanity"
sudo ip neigh flush dev bond0 2>/dev/null
for i in /sys/class/net/thunderbolt*; do
  n=$(basename "$i")
  echo "  $n carrier=$(cat $i/carrier 2>/dev/null) rx=$(cat $i/statistics/rx_packets 2>/dev/null)"
done
echo "done. Now run odl-bringup.sh on BOTH nodes."
