#!/bin/bash
set -e
# provide a virtual libtinfo-dev (runtime libtinfo.so.6 is present; comgr only needs the .so)
sudo apt-get install -y equivs 2>&1 | tail -1
cd /tmp
cat > libtinfo-dev-stub <<CTRL
Section: misc
Priority: optional
Standards-Version: 3.9.2
Package: libtinfo-dev
Version: 6.6
Provides: libtinfo-dev
Description: stub satisfying ROCm comgr dep; runtime libtinfo.so.6 already present
CTRL
equivs-build libtinfo-dev-stub 2>&1 | tail -2
sudo dpkg -i libtinfo-dev_6.6_all.deb 2>&1 | tail -1
echo '=== retry rocm-hip-runtime ==='
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y rocm-hip-runtime 2>&1 | tail -8
echo "INSTALL2_DONE_$?"
ls -d /opt/rocm-7.2.0 2>/dev/null && echo "  /opt/rocm-7.2.0 present" || echo "  MISSING"
