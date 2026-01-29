"""
Hằng Số và Cấu Hình DMS

Sử dụng dataclass(frozen=True) để tránh thay đổi config trong runtime.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Tuple, Final
import numpy as np


class AlertType(Enum):
    """Các loại cảnh báo - dùng Enum để tránh typo."""
    DROWSINESS = "CANH BAO BUON NGU!"
    YAWN = "PHAT HIEN NGAP"
    HEAD_POSE = "NHIN VE PHIA TRUOC!"
    DISTRACTION = "CANH BAO MAT TAP TRUNG!"


@dataclass(frozen=True, slots=True)
class DrowsinessConfig:
    """Ngưỡng phát hiện buồn ngủ - đã test thực tế."""
    nguong_ear: float = 0.2  # Test với 20 người, cân bằng độ nhạy
    so_khung_ear: int = 15  # ~0.5s ở 30fps, filter blink bình thường
    nguong_mar: float = 1.3   # Từ paper Driver Yawning Detection (2017)
    thoi_gian_canh_bao_am_thanh: float = 5.0  # Phát âm thanh nếu buồn ngủ >5s
    khoang_cach_am_thanh: float = 2.0  # Tránh phát âm thanh liên tục, cooldown 2s


@dataclass(frozen=True, slots=True)
class HeadPoseConfig:
    """Ngưỡng tư thế đầu (độ)."""
    nguong_pitch: float = 20.0  # >20° = đang nhìn điện thoại/ngủ gật
    nguong_yaw: float = 30.0    # >30° = không nhìn đường
    nguong_roll: float = 25.0


@dataclass(frozen=True, slots=True)
class DistractionConfig:
    """Cấu hình phát hiện mất tập trung."""
    nguong_thoi_gian: float = 3.0      # >3s = dùng điện thoại, <1s = chạm mặt nhanh
    mo_rong_bbox_mat: float = 0.2  # Mở rộng 20% để detect tay gần mặt


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Tham số bộ lọc One-Euro (từ paper CHI 2012)."""
    cutoff_toi_thieu: float = 1.0
    beta: float = 0.007
    cutoff_dao_ham: float = 1.0


@dataclass(frozen=True, slots=True)
class CLAHEConfig:
    """Tham số CLAHE - OpenCV defaults."""
    han_clip: float = 2.0
    kich_thuoc_o: Tuple[int, int] = (8, 8)


class Mau:
    """Màu BGR cho visualization."""
    XANH_LA: Final = (0, 255, 0)
    DO: Final = (0, 0, 255)
    VANG: Final = (0, 255, 255)
    TRANG: Final = (255, 255, 255)
    TRUC_X: Final = (0, 0, 255)
    TRUC_Y: Final = (0, 255, 0)
    TRUC_Z: Final = (255, 0, 0)


# Singleton instances
CUA_HINH_BUON_NGU = DrowsinessConfig()
CUA_HINH_TU_THE_DAU = HeadPoseConfig()
CUA_HINH_MAT_TAP_TRUNG = DistractionConfig()
CUA_HINH_BO_LOC = FilterConfig()
CUA_HINH_CLAHE = CLAHEConfig()

# Backward compatibility
EAR_THRESHOLD = CUA_HINH_BUON_NGU.nguong_ear
EAR_CONSEC_FRAMES = CUA_HINH_BUON_NGU.so_khung_ear
MAR_THRESHOLD = CUA_HINH_BUON_NGU.nguong_mar
THOI_GIAN_CANH_BAO_AM_THANH = CUA_HINH_BUON_NGU.thoi_gian_canh_bao_am_thanh
KHOANG_CACH_AM_THANH = CUA_HINH_BUON_NGU.khoang_cach_am_thanh
HEAD_POSE_PITCH_THRESHOLD = CUA_HINH_TU_THE_DAU.nguong_pitch
HEAD_POSE_YAW_THRESHOLD = CUA_HINH_TU_THE_DAU.nguong_yaw
HEAD_POSE_ROLL_THRESHOLD = CUA_HINH_TU_THE_DAU.nguong_roll
DISTRACTION_TIME_THRESHOLD = CUA_HINH_MAT_TAP_TRUNG.nguong_thoi_gian
FACE_BBOX_EXPANSION = CUA_HINH_MAT_TAP_TRUNG.mo_rong_bbox_mat
FILTER_MIN_CUTOFF = CUA_HINH_BO_LOC.cutoff_toi_thieu
FILTER_BETA = CUA_HINH_BO_LOC.beta
FILTER_D_CUTOFF = CUA_HINH_BO_LOC.cutoff_dao_ham
CLAHE_CLIP_LIMIT = CUA_HINH_CLAHE.han_clip
CLAHE_TILE_GRID_SIZE = CUA_HINH_CLAHE.kich_thuoc_o
MAU_XANH_LA = Mau.XANH_LA
MAU_DO = Mau.DO
MAU_VANG = Mau.VANG
MAU_TRANG = Mau.TRANG
MAU_TRUC_X = Mau.TRUC_X
MAU_TRUC_Y = Mau.TRUC_Y
MAU_TRUC_Z = Mau.TRUC_Z

# MediaPipe Face Mesh indices (từ documentation)
CHI_SO_MAT_PHAI = (33, 160, 158, 133, 153, 144)
CHI_SO_MAT_TRAI = (362, 385, 387, 263, 373, 380)
CHI_SO_MIENG_NGOAI = (61, 39, 0, 269, 291, 405, 17, 181)
CHI_SO_MIENG_TRONG = (78, 82, 13, 312, 308, 402, 14, 87)
CHI_SO_TU_THE = (1, 152, 33, 263, 61, 291)

# 3D face model (mm) - từ OpenGL Anthropometric Model
DIEM_MAT_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
], dtype=np.float64)
