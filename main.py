"""
Hệ Thống Giám Sát Tài Xế (DMS)

Sử dụng: python main.py [--camera 0] [--width 640] [--height 480]
Nhấn 'q' để thoát.
"""

from __future__ import annotations
import argparse
import logging
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


def phat_am_thanh_async(duong_dan: str) -> None:
    """Phát âm thanh trong thread riêng để không block."""
    try:
        import playsound
        playsound.playsound(duong_dan, block=False)
        logger.info(f"Phát âm thanh: {duong_dan}")
    except ImportError:
        logger.warning("Chưa cài playsound. Cài: pip install playsound")
    except Exception as e:
        logger.warning(f"Lỗi phát âm thanh: {e}")


@dataclass
class CauHinhCamera:
    id_camera: int = 0
    chieu_rong: int = 640
    chieu_cao: int = 480
    fps: int = 30


@contextmanager
def mo_camera(cau_hinh: CauHinhCamera) -> Generator[cv2.VideoCapture, None, None]:
    may_quay = cv2.VideoCapture(cau_hinh.id_camera)
    may_quay.set(cv2.CAP_PROP_FRAME_WIDTH, cau_hinh.chieu_rong)
    may_quay.set(cv2.CAP_PROP_FRAME_HEIGHT, cau_hinh.chieu_cao)
    may_quay.set(cv2.CAP_PROP_FPS, cau_hinh.fps)
    if not may_quay.isOpened():
        raise RuntimeError(f"Không thể mở camera {cau_hinh.id_camera}")
    logger.info(f"Camera sẵn sàng ({cau_hinh.chieu_rong}x{cau_hinh.chieu_cao}@{cau_hinh.fps}fps)")
    try:
        yield may_quay
    finally:
        may_quay.release()


@dataclass
class ThongKeFPS:
    """Moving average FPS."""
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


@dataclass
class HeThongGiamSatTaiXe:
    cau_hinh_camera: CauHinhCamera = field(default_factory=CauHinhCamera)
    ten_cua_so: str = "He Thong Giam Sat Tai Xe"
    duong_dan_am_thanh: str = "chiken-on-tree.mp3"
    
    _tien_xu_ly: TienXuLyCLAHE = field(init=False, repr=False)
    _phan_tich_mat: PhanTichMat = field(init=False, repr=False)
    _theo_doi_tay: TheoDoiTay = field(init=False, repr=False)
    _trao_dua_tinh_nang: TraoDuaTinhNang = field(init=False, repr=False)
    _fps: ThongKeFPS = field(init=False, repr=False)
    
    # Tracking buồn ngủ
    _thoi_gian_buon_ngu_bat_dau: Optional[float] = field(default=None, repr=False)
    _thoi_gian_am_thanh_cuoi: float = field(default=0.0, repr=False)
    
    def __post_init__(self) -> None:
        logger.info("Khởi tạo DMS...")
        self._tien_xu_ly = TienXuLyCLAHE()
        self._phan_tich_mat = PhanTichMat()
        self._theo_doi_tay = TheoDoiTay()
        self._trao_dua_tinh_nang = TraoDuaTinhNang()
        self._fps = ThongKeFPS()
        logger.info("DMS sẵn sàng!")
    
    def chay(self) -> None:
        logger.info("Đang chạy... Nhấn 'q' để thoát.")
        with mo_camera(self.cau_hinh_camera) as may_quay:
            while True:
                thanh_cong, khung_hinh = may_quay.read()
                if not thanh_cong:
                    break
                dau_ra = self._xu_ly(khung_hinh)
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
        
        # ========== TRACKING BUỒN NGỦ ==========
        if ket_qua_mat['canh_bao_buon_ngu']:
            if self._thoi_gian_buon_ngu_bat_dau is None:
                self._thoi_gian_buon_ngu_bat_dau = ts
            else:
                thoi_gian_buon_ngu = ts - self._thoi_gian_buon_ngu_bat_dau
                # Phát âm thanh nếu buồn ngủ >5s và cooldown đã hết
                if thoi_gian_buon_ngu >= THOI_GIAN_CANH_BAO_AM_THANH and \
                   (ts - self._thoi_gian_am_thanh_cuoi) >= KHOANG_CACH_AM_THANH:
                    luong = threading.Thread(
                        target=phat_am_thanh_async, 
                        args=(self.duong_dan_am_thanh,),
                        daemon=True
                    )
                    luong.start()
                    self._thoi_gian_am_thanh_cuoi = ts
                    logger.warning(f"⚠️ CẢNH BÁO BUỒN NGỦ! Thời gian: {thoi_gian_buon_ngu:.1f}s")
        else:
            self._thoi_gian_buon_ngu_bat_dau = None
        
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
                                     ket_qua_mat['roll'], fps)
        dau_ra = self._trao_dua_tinh_nang.ve_canh_bao(dau_ra, ket_qua_mat['canh_bao_buon_ngu'],
                                    ket_qua_mat['canh_bao_ngap'], ket_qua_mat['canh_bao_tu_the'],
                                    ket_qua_tay['distraction_alert'])
        return dau_ra
    
    def _dung(self) -> None:
        self._phan_tich_mat.release()
        self._theo_doi_tay.release()
        cv2.destroyAllWindows()
        logger.info("Đã tắt DMS.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Driver Monitoring System")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--width", "-W", type=int, default=640)
    parser.add_argument("--height", "-H", type=int, default=480)
    args = parser.parse_args()
    
    try:
        cau_hinh = CauHinhCamera(args.camera, args.width, args.height)
        HeThongGiamSatTaiXe(cau_hinh).chay()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
