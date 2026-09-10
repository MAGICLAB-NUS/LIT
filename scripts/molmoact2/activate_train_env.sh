#!/usr/bin/env bash
# Fast env: reuse packages from molmo_serious lerobot/.venv, code from clean molmoact2/lerobot
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/env.sh"
export PATH="${WS}/lerobot/.venv/bin:${PATH}"
export VIRTUAL_ENV="${WS}/lerobot/.venv"
hash -r
echo "[activate] python=$(command -v python)"
echo "[activate] lerobot-train=$(command -v lerobot-train)"
python -c 'import pathlib,lerobot; print("[activate] lerobot ->", pathlib.Path(lerobot.__file__).resolve())'
