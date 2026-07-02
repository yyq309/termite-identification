#!/usr/bin/env bash
# Offline provisioning of the air-gapped RTX 4090 server.
# Assumes this bundle was scp'd to $ROOT with:
#   $ROOT/wheels/*.whl                     (cp38 / linux_x86_64 / cu121 torch + deps)
#   $ROOT/python/bin/python3               (self-contained CPython 3.8 w/ pip, from
#                                           python-build-standalone; the system python3
#                                           is Debian-stripped: no pip/distutils/ensurepip)
# Installs everything with NO network access, then verifies CUDA on the GPUs.
set -euo pipefail

ROOT="${1:-/root/termite}"
cd "$ROOT"
PYBIN="${PYBIN:-$ROOT/python/bin/python3}"
echo "== python: $($PYBIN --version) =="

echo "== install stack (offline) =="
"$PYBIN" -m pip install --no-index --find-links wheels \
    torch torchvision ultralytics scikit-learn
# The bare container has no libGL, so the GUI opencv-python (pulled in by ultralytics)
# would fail `import cv2`. Force the headless build to own cv2 (installed last wins).
"$PYBIN" -m pip install --no-index --find-links wheels --force-reinstall --no-deps \
    opencv-python-headless

echo "== verify torch + CUDA =="
"$PYBIN" - <<'PY'
import torch, torchvision, ultralytics, cv2
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv     ", cv2.__version__)
print("cuda avail ", torch.cuda.is_available())
print("device cnt ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}", torch.cuda.get_device_name(i))
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok  ", float((x @ x).sum()) != 0)
PY
echo "== done =="
