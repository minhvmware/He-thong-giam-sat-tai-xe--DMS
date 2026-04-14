from __future__ import annotations
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional
import cv2
import numpy as np
from dms.preprocessing import TienXuLyCLAHE
from dms.face_analysis import PhanTichMat
from dms.hand_tracking import TheoDoiTay
from dms.visualization import TraoDuaTinhNang
from dms.constants import THOI_GIAN_CANH_BAO_AM_THANH, KHOANG_CACH_AM_THANH
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_web_frame_lock = threading.Lock()
_latest_web_frame = None

def chay_web_server():
    try:
        from flask import Flask, Response
        app = Flask(__name__)
        @app.route('/')
        def index():
            html = "<html><body style='margin:0;background:#000;text-align:center;'>" \
                   "<h2 style='color:#fff;font-family:sans-serif;'>DMS Live View</h2>" \
                   "<img src='/video_feed' style='max-width:100%; max-height:90vh;'></body></html>"
            return html
        def gen():
            global _latest_web_frame
            while True:
                with _web_frame_lock:
                    frame = _latest_web_frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        @app.route('/video_feed')
        def video_feed():
            return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        logger.info("Khởi động Web Stream tại http://<IP_cua_Pi>:5000/")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except ImportError:
        logger.warning("Chưa cài Flask. (Hãy sửa lỗi bằng: pip install flask)")
def liet_ke_thiet_bi_am_thanh() -> list[str]:
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=5
        )
        lines = []
        for line in result.stdout.splitlines():
            if line.startswith("card "):
                lines.append(line.strip())
        return lines
    except Exception:
        return []
def _phat_wav_alsa(duong_dan: str, device: Optional[str] = None) -> None:
    cmd = ["aplay"]
    if device:
        cmd.extend(["-D", device])
    cmd.append(duong_dan)
    subprocess.run(cmd, stderr=subprocess.DEVNULL, timeout=30)
def _phat_mp3_pygame(duong_dan: str, device: Optional[str] = None) -> None:
    try:
        import pygame
        if not pygame.mixer.get_init():
            if device:
                os.environ["SDL_AUDIODRIVER"] = "alsa"
                os.environ["AUDIODEV"] = device
            pygame.mixer.init()
        pygame.mixer.music.load(duong_dan)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        logger.warning(f"pygame lỗi: {e}")
def phat_am_thanh(duong_dan: str, device: Optional[str] = None) -> None:
    try:
        if duong_dan.endswith(".wav"):
            _phat_wav_alsa(duong_dan, device)
        else:
            _phat_mp3_pygame(duong_dan, device)
        logger.info(f"Phát âm thanh: {duong_dan}")
    except Exception as e:
        logger.warning(f"Lỗi phát âm thanh: {e}")
def phat_tts(van_ban: str, device: Optional[str] = None, ngon_ngu: str = "vi") -> None:
    if not shutil.which("espeak-ng"):
        logger.warning("espeak-ng chưa cài. Cài: sudo apt install espeak-ng")
        return
    try:
        if device:
            p1 = subprocess.Popen(
                ["espeak-ng", "-v", ngon_ngu, "--stdout", van_ban],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            p2 = subprocess.Popen(
                ["aplay", "-D", device],
                stdin=p1.stdout, stderr=subprocess.DEVNULL
            )
            p1.stdout.close()
            p2.wait(timeout=10)
        else:
            subprocess.run(
                ["espeak-ng", "-v", ngon_ngu, van_ban],
                stderr=subprocess.DEVNULL, timeout=10
            )
        logger.info(f"TTS: {van_ban}")
    except Exception as e:
        logger.warning(f"TTS lỗi: {e}")
@dataclass
class CauHinhCamera:
    id_camera: int = 0
    chieu_rong: int = 640
    chieu_cao: int = 480
    fps: int = 30
class CameraCapture:
    def __init__(self, cau_hinh: CauHinhCamera):
        self._picam2 = None
        self._rpicam_proc = None
        self._cv2_cap = None
        self._width = cau_hinh.chieu_rong
        self._height = cau_hinh.chieu_cao
        self._yuv_frame_size = self._width * self._height * 3 // 2
        try:
            from picamera2 import Picamera2
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(
                main={"size": (self._width, self._height), "format": "RGB888"}
            )
            self._picam2.configure(config)
            self._picam2.start()
            logger.info(f"Camera CSI sẵn sàng (picamera2) ({self._width}x{self._height})")
            return
        except (ImportError, ModuleNotFoundError):
            logger.info("picamera2/libcamera không có, thử rpicam-vid...")
        except Exception as e:
            logger.warning(f"picamera2 lỗi: {e}, thử rpicam-vid...")
            self._picam2 = None
        if shutil.which("rpicam-vid"):
            try:
                self._rpicam_proc = subprocess.Popen(
                    [
                        "rpicam-vid", "-t", "0",
                        "--width", str(self._width),
                        "--height", str(self._height),
                        "--framerate", str(cau_hinh.fps),
                        "--codec", "yuv420",
                        "--nopreview", "-o", "-"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                test_data = self._rpicam_proc.stdout.read(self._yuv_frame_size)
                if len(test_data) == self._yuv_frame_size:
                    self._first_frame = test_data
                    logger.info(f"Camera CSI sẵn sàng (rpicam-vid) "
                                f"({self._width}x{self._height}@{cau_hinh.fps}fps)")
                    return
                else:
                    raise RuntimeError("rpicam-vid không trả về frame hợp lệ")
            except Exception as e:
                logger.warning(f"rpicam-vid lỗi: {e}, thử OpenCV...")
                if self._rpicam_proc:
                    self._rpicam_proc.terminate()
                    self._rpicam_proc = None
        self._cv2_cap = cv2.VideoCapture(cau_hinh.id_camera)
        self._cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cv2_cap.set(cv2.CAP_PROP_FPS, cau_hinh.fps)
        if not self._cv2_cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {cau_hinh.id_camera}")
        logger.info(f"Camera USB sẵn sàng (OpenCV) "
                    f"({self._width}x{self._height}@{cau_hinh.fps}fps)")
    def read(self):
        if self._picam2:
            frame = self._picam2.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._rpicam_proc:
            if hasattr(self, '_first_frame') and self._first_frame is not None:
                data = self._first_frame
                self._first_frame = None
            else:
                data = self._rpicam_proc.stdout.read(self._yuv_frame_size)
            if len(data) != self._yuv_frame_size:
                return False, None
            yuv = np.frombuffer(data, dtype=np.uint8).reshape(
                (self._height * 3 // 2, self._width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        return self._cv2_cap.read()
    def release(self):
        if self._picam2:
            self._picam2.close()
            self._picam2 = None
        if self._rpicam_proc:
            self._rpicam_proc.terminate()
            self._rpicam_proc.wait(timeout=3)
            self._rpicam_proc = None
        if self._cv2_cap:
            self._cv2_cap.release()
            self._cv2_cap = None
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.release()
@contextmanager
def mo_camera(cau_hinh: CauHinhCamera) -> Generator[CameraCapture, None, None]:
    may_quay = CameraCapture(cau_hinh)
    try:
        yield may_quay
    finally:
        may_quay.release()
@dataclass
class ThongKeFPS:
    cua_so: int = 30
    _lich_su: list = field(default_factory=list, repr=False)
    _truoc: float = field(default_factory=time.time, repr=False)
    def cap_nhat(self) -> float:
        bay_gio = time.time()
        self._lich_su.append(1.0 / max(bay_gio - self._truoc, 1e-6))
        self._truoc = bay_gio
        if len(self._lich_su) > self.cua_so:
            self._lich_su.pop(0)
        return sum(self._lich_su) / len(self._lich_su)
TTS_CANH_BAO = {
    'buon_ngu': "Cảnh báo! Bạn đang buồn ngủ! Hãy dừng xe nghỉ ngơi!",
    'ngap': "Bạn đang ngáp, hãy nghỉ ngơi!",
    'tu_the': "Hãy nhìn thẳng về phía trước!",
    'mat_tap_trung': "Tập trung lái xe!",
    'con': "Cảnh báo! Phát hiện nồng độ cồn! Vui lòng tắt máy và không lái xe!",
}
@dataclass
class HeThongGiamSatTaiXe:
    cau_hinh_camera: CauHinhCamera = field(default_factory=CauHinhCamera)
    ten_cua_so: str = "He Thong Giam Sat Tai Xe"
    duong_dan_am_thanh: str = "chiken-on-tree.mp3"
    headless: bool = False
    tts_bat: bool = False
    audio_device: Optional[str] = None
    web_stream: bool = False
    mq3_pin: Optional[int] = None
    _tien_xu_ly: TienXuLyCLAHE = field(init=False, repr=False)
    _phan_tich_mat: PhanTichMat = field(init=False, repr=False)
    _theo_doi_tay: TheoDoiTay = field(init=False, repr=False)
    _trao_dua_tinh_nang: TraoDuaTinhNang = field(init=False, repr=False)
    _fps: ThongKeFPS = field(init=False, repr=False)
    _thoi_gian_buon_ngu_bat_dau: Optional[float] = field(default=None, repr=False)
    _thoi_gian_am_thanh_cuoi: float = field(default=0.0, repr=False)
    _tts_bat_dau: dict = field(default_factory=dict, repr=False)
    _tts_da_phat: dict = field(default_factory=dict, repr=False)
    _TTS_DELAY: float = 3.0  
    _TTS_COOLDOWN: float = 8.0  
    _cam_bien_con: object = field(init=False, default=None, repr=False)
    _co_con: bool = field(init=False, default=False, repr=False)
    def __post_init__(self) -> None:
        logger.info("Khởi tạo DMS...")
        self._tien_xu_ly = TienXuLyCLAHE()
        self._phan_tich_mat = PhanTichMat()
        self._theo_doi_tay = TheoDoiTay()
        self._trao_dua_tinh_nang = TraoDuaTinhNang()
        self._fps = ThongKeFPS()
        mode = "headless" if self.headless else "GUI"
        tts = "TTS bật" if self.tts_bat else "TTS tắt"
        audio = self.audio_device or "mặc định"
        logger.info(f"DMS sẵn sàng! [{mode}] [{tts}] [Loa: {audio}]")

        if self.mq3_pin is not None:
            try:
                from gpiozero import DigitalInputDevice
                self._cam_bien_con = DigitalInputDevice(self.mq3_pin)
                def doc_con():
                    while True:
                        self._co_con = not getattr(self._cam_bien_con, 'is_active', True)
                        time.sleep(0.5)
                threading.Thread(target=doc_con, daemon=True).start()
                logger.info(f"Đã kích hoạt cảm biến MQ-3 (nồng độ cồn) tại GPIO {self.mq3_pin}")
            except ImportError:
                logger.warning(f"Chưa cài gpiozero. Bỏ qua cảm biến GPIO {self.mq3_pin}.")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo GPIO {self.mq3_pin} cho cảm biến cồn: {e}")

    def _canh_bao_tts(self, loai: str) -> None:
        if not self.tts_bat:
            return
        ts = time.time()
        if loai not in self._tts_bat_dau:
            self._tts_bat_dau[loai] = ts
            self._tts_da_phat[loai] = False
            return
        thoi_gian = ts - self._tts_bat_dau[loai]
        if thoi_gian < self._TTS_DELAY:
            return
        if self._tts_da_phat.get(loai, False):
            if thoi_gian >= self._TTS_DELAY + self._TTS_COOLDOWN:
                self._tts_bat_dau[loai] = ts
                self._tts_da_phat[loai] = False
            return
        self._tts_da_phat[loai] = True
        van_ban = TTS_CANH_BAO.get(loai, "")
        if van_ban:
            t = threading.Thread(
                target=phat_tts,
                args=(van_ban,),
                kwargs={"device": self.audio_device},
                daemon=True
            )
            t.start()
            logger.info(f"🔊 TTS [{loai}]: {van_ban}")
    def _reset_tts(self, loai: str) -> None:
        self._tts_bat_dau.pop(loai, None)
        self._tts_da_phat.pop(loai, None)
    def chay(self) -> None:
        if self.headless:
            logger.info("Chế độ headless — không hiển thị cửa sổ. Ctrl+C để tắt.")
        else:
            logger.info("Đang chạy... Nhấn 'q' để thoát.")
        if self.web_stream:
            threading.Thread(target=chay_web_server, daemon=True).start()
        with mo_camera(self.cau_hinh_camera) as may_quay:
            while True:
                thanh_cong, khung_hinh = may_quay.read()
                if not thanh_cong:
                    logger.warning("Không đọc được frame, thử lại...")
                    time.sleep(0.1)
                    continue
                dau_ra = self._xu_ly(khung_hinh)

                if self.web_stream:
                    thanh_cong_encode, buffer = cv2.imencode('.jpg', dau_ra)
                    if thanh_cong_encode:
                        global _latest_web_frame
                        with _web_frame_lock:
                            _latest_web_frame = buffer.tobytes()

                if not self.headless:
                    cv2.imshow(self.ten_cua_so, dau_ra)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        self._dung()
    def _xu_ly(self, khung_hinh: np.ndarray) -> np.ndarray:
        ts = time.time()
        anh_tang_cuong = self._tien_xu_ly.tang_cuong(khung_hinh)
        ket_qua_mat = self._phan_tich_mat.analyze(anh_tang_cuong, ts)
        ket_qua_tay = self._theo_doi_tay.analyze(anh_tang_cuong, ket_qua_mat.get('khung_bbox_mat'))
        fps = self._fps.cap_nhat()
        if ket_qua_mat['canh_bao_buon_ngu']:
            self._canh_bao_tts('buon_ngu')
            if self._thoi_gian_buon_ngu_bat_dau is None:
                self._thoi_gian_buon_ngu_bat_dau = ts
            else:
                thoi_gian_buon_ngu = ts - self._thoi_gian_buon_ngu_bat_dau
                if thoi_gian_buon_ngu >= THOI_GIAN_CANH_BAO_AM_THANH and                   (ts - self._thoi_gian_am_thanh_cuoi) >= KHOANG_CACH_AM_THANH:
                    luong = threading.Thread(
                        target=phat_am_thanh,
                        args=(self.duong_dan_am_thanh,),
                        kwargs={"device": self.audio_device},
                        daemon=True
                    )
                    luong.start()
                    self._thoi_gian_am_thanh_cuoi = ts
                    logger.warning(f"⚠️ CẢNH BÁO BUỒN NGỦ! Thời gian: {thoi_gian_buon_ngu:.1f}s")
        else:
            self._thoi_gian_buon_ngu_bat_dau = None
            self._reset_tts('buon_ngu')
        if ket_qua_mat['canh_bao_ngap']:
            self._canh_bao_tts('ngap')
        else:
            self._reset_tts('ngap')
        if ket_qua_tay['distraction_alert']:
            self._canh_bao_tts('mat_tap_trung')
        else:
            self._reset_tts('mat_tap_trung')
        if self._co_con:
            self._canh_bao_tts('con')
        else:
            self._reset_tts('con')
        dau_ra = anh_tang_cuong.copy()
        if ket_qua_mat['mat_phat_hien']:
            dau_ra = self._trao_dua_tinh_nang.ve_luoi_mat(dau_ra, ket_qua_mat['diem_moc'])
            if ket_qua_mat['vec_quay'] is not None:
                diem_moc = ket_qua_mat['diem_moc']
                chieu_cao, chieu_rong = dau_ra.shape[:2]
                dau_ra = self._trao_dua_tinh_nang.ve_truc_tu_the_dau(
                    dau_ra, ket_qua_mat['vec_quay'], ket_qua_mat['vec_tuan'],
                    (diem_moc[1].x*chieu_rong, diem_moc[1].y*chieu_cao))
        dau_ra = self._trao_dua_tinh_nang.ve_diem_moc_tay(dau_ra, ket_qua_tay['hand_landmarks'])
        dau_ra = self._trao_dua_tinh_nang.ve_so_lieu(dau_ra, ket_qua_mat['ear'], ket_qua_mat['mar'],
                                     ket_qua_mat['pitch'], ket_qua_mat['yaw'],
                                     ket_qua_mat['roll'], fps, self._co_con)
        dau_ra = self._trao_dua_tinh_nang.ve_canh_bao(dau_ra, ket_qua_mat['canh_bao_buon_ngu'],
                                    ket_qua_mat['canh_bao_ngap'], False,
                                    ket_qua_tay['distraction_alert'], self._co_con)
        return dau_ra
    def _dung(self) -> None:
        self._phan_tich_mat.release()
        self._theo_doi_tay.release()
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Đã tắt DMS.")
def main() -> int:
    parser = argparse.ArgumentParser(description="Hệ Thống Giám Sát Tài Xế (DMS)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="ID camera (mặc định: 0)")
    parser.add_argument("--width", "-W", type=int, default=640)
    parser.add_argument("--height", "-H", type=int, default=480)
    parser.add_argument("--headless", action="store_true",
                        help="Chạy không hiển thị (dùng cho auto-start)")
    parser.add_argument("--tts", action="store_true",
                        help="Bật cảnh báo giọng nói (espeak-ng)")
    parser.add_argument("--web", action="store_true",
                        help="Phát Web Stream tại cổng 5000 để điện thoại canh góc nhìn")
    parser.add_argument("--audio-device", type=str, default=None,
                        help="Thiết bị âm thanh ALSA (VD: plughw:1,0)")
    parser.add_argument("--list-audio", action="store_true",
                        help="Liệt kê thiết bị âm thanh rồi thoát")
    parser.add_argument("--mq3-pin", type=int, default=None,
                        help="Chân GPIO cắm mạch MQ-3 đo nồng độ cồn (VD: 4)")
    args = parser.parse_args()
    if args.list_audio:
        print("=== Thiết bị âm thanh ===")
        devices = liet_ke_thiet_bi_am_thanh()
        if devices:
            for d in devices:
                print(f"  {d}")
            print("\nDùng: --audio-device plughw:<card>,<device>")
        else:
            print("  Không tìm thấy thiết bị nào.")
        return 0
    try:
        cau_hinh = CauHinhCamera(args.camera, args.width, args.height)
        dms = HeThongGiamSatTaiXe(
            cau_hinh_camera=cau_hinh,
            headless=args.headless,
            tts_bat=args.tts,
            audio_device=args.audio_device,
            web_stream=args.web,
            mq3_pin=args.mq3_pin
        )
        dms.chay()
        return 0
    except KeyboardInterrupt:
        logger.info("Tắt bởi Ctrl+C")
        return 0
    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        return 1
if __name__ == "__main__":
    sys.exit(main())
