#!/bin/bash
# Script khởi động DMS trên Raspberry Pi (GUI mode)
# Tự chạy khi desktop load

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/dms-env"

# Kích hoạt virtual environment
source "$VENV_DIR/bin/activate"

# Đợi camera + desktop sẵn sàng
sleep 5

# Chạy DMS với GUI + TTS
cd "$SCRIPT_DIR"
exec python main.py --tts "$@"
