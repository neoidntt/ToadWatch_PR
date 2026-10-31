#!/usr/bin/env bash
# Double-click me (or run: ./install_libraries.sh): installs every library ToadWatch_PR needs.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
echo "=== ToadWatch_PR: installing libraries ==="
if ! command -v python3 >/dev/null; then
  echo "Python 3 is not installed. Install it first (Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip)."; read -rp "Press Enter to close"; exit 1
fi
PY=python3
if ! python3 -m venv .venv 2>/dev/null; then
  rm -rf .venv
  if command -v apt-get >/dev/null; then
    echo "Installing python3-venv (your password may be asked)..."
    sudo apt-get install -y python3-venv python3-pip && python3 -m venv .venv 2>/dev/null
  fi
fi
if [ -x .venv/bin/python ]; then
  . .venv/bin/activate; PY=python
  $PY -m pip install --upgrade pip
  $PY -m pip install -r requirements.txt || { echo "Installation failed."; read -rp "Press Enter to close"; exit 1; }
else
  echo "No virtual environment possible; installing for your user instead."
  $PY -m pip install --user --break-system-packages -r requirements.txt || { echo "Installation failed."; read -rp "Press Enter to close"; exit 1; }
fi
$PY -m ipykernel install --user --name toadwatch --display-name "ToadWatch (.venv)"
command -v soffice >/dev/null || echo "NOTE: LibreOffice is not installed; the reports are still made as Word/PowerPoint but the PDFs need it (sudo apt install libreoffice)."
echo
echo "Done. In VS Code open this folder, pick the kernel 'ToadWatch (.venv)' (or your default Python) and run notebooks/00_START_HERE.ipynb."
read -rp "Press Enter to close"
