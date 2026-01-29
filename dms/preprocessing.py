"""
CLAHE Preprocessing - Cải thiện ảnh ánh sáng yếu

CLAHE tốt hơn histogram equalization toàn cục vì:
- Xử lý local, không bị artifacts
- Có giới hạn contrast để tránh khuếch đại noise
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np
from .constants import CUA_HINH_CLAHE


class KhongGianMau(Enum):
    YCRCB = "ycrcb"  # Nhanh hơn
    LAB = "lab"      # Chính xác hơn về tri giác


@dataclass
class TienXuLyCLAHE:
    """Tiền xử lý CLAHE cho low-light."""
    han_clip: float = field(default_factory=lambda: CUA_HINH_CLAHE.han_clip)
    kich_thuoc_o: tuple = field(default_factory=lambda: CUA_HINH_CLAHE.kich_thuoc_o)
    khong_gian_mau: KhongGianMau = field(default=KhongGianMau.YCRCB)
    _clahe: cv2.CLAHE = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=self.han_clip,
            tileGridSize=self.kich_thuoc_o
        )
        
    def tang_cuong(self, khung_hinh: np.ndarray) -> np.ndarray:
        """Chỉ xử lý kênh luminance để giữ màu."""
        if khung_hinh is None or khung_hinh.size == 0:
            return khung_hinh
        
        if self.khong_gian_mau == KhongGianMau.LAB:
            chuan_hoa = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2LAB)
            danh_sach_kenh = list(cv2.split(chuan_hoa))
            danh_sach_kenh[0] = self._clahe.apply(danh_sach_kenh[0])
            return cv2.cvtColor(cv2.merge(danh_sach_kenh), cv2.COLOR_LAB2BGR)
        else:
            chuan_hoa = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2YCrCb)
            danh_sach_kenh = list(cv2.split(chuan_hoa))
            danh_sach_kenh[0] = self._clahe.apply(danh_sach_kenh[0])
            return cv2.cvtColor(cv2.merge(danh_sach_kenh), cv2.COLOR_YCrCb2BGR)
