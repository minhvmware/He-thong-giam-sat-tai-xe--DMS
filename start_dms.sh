#!/bin/bash
# Script khởi động DMS trên Raspberry Pi (GUI mode)
# Tự chạy khi desktop load

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/dms-env"

# Kích hoạt virtual environment
source "$VENV_DIR/bin/activate"

# Đặt DISPLAY để GUI hiện trên màn hình (Wayland hoặc X11)
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# Đợi desktop + camera sẵn sàng
sleep 8

# Chạy DMS với GUI + TTS
cd "$SCRIPT_DIR"
exec python main.py --tts --mq3-pin 17 "$@"
