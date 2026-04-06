#!/bin/bash
# Script khởi động DMS trên Raspberry Pi
# Đặt tại: /home/minh/He-thong-giam-sat-tai-xe--DMS/start_dms.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/dms-env"

# Kích hoạt virtual environment
source "$VENV_DIR/bin/activate"

# Đợi camera sẵn sàng (3 giây sau boot)
sleep 3

# Chạy DMS headless + TTS
cd "$SCRIPT_DIR"
exec python main.py --headless --tts "$@"
