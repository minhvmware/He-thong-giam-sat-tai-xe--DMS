"""
Hand Tracking - Phát hiện mất tập trung

Phát hiện khi tay ở gần mặt >3s (dùng điện thoại, ăn uống).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, List, NamedTuple
import cv2
import numpy as np
import mediapipe as mp
from .constants import CUA_HINH_MAT_TAP_TRUNG


class Diem2D(NamedTuple):
    x: float
    y: float


class KhungBbox(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    def thua_chua(self, diem: Diem2D, mo_rong: float = 0.0) -> bool:
        w, h = self.x_max - self.x_min, self.y_max - self.y_min
        return (
            self.x_min - w*mo_rong <= diem.x <= self.x_max + w*mo_rong and
            self.y_min - h*mo_rong <= diem.y <= self.y_max + h*mo_rong
        )


@dataclass
class KetQuaTheoDoiTay:
    so_tay_phat_hien: int = 0
    diem_moc_tay: List = field(default_factory=list)
    khung_bbox_tay: List = field(default_factory=list)
    tay_gan_mat: bool = False
    canh_bao_mat_tap_trung: bool = False
    thoi_gian_mat_tap_trung: float = 0.0
    
    def thanh_dict(self) -> dict:
        return {
            'hands_detected': self.so_tay_phat_hien,
            'hand_landmarks': self.diem_moc_tay,
            'hand_bboxes': [{'x_min': int(b.x_min), 'x_max': int(b.x_max),
                            'y_min': int(b.y_min), 'y_max': int(b.y_max)} 
                           for b in self.khung_bbox_tay],
            'hand_near_face': self.tay_gan_mat,
            'distraction_alert': self.canh_bao_mat_tap_trung,
            'distraction_duration': self.thoi_gian_mat_tap_trung
        }


@dataclass
class TheoDoiTay:
    so_tay_toi_da: int = 2
    do_tin_cay_phat_hien: float = 0.5
    do_tin_cay_theo_doi: float = 0.5
    nguong_thoi_gian: float = field(default_factory=lambda: CUA_HINH_MAT_TAP_TRUNG.nguong_thoi_gian)
    mo_rong_bbox: float = field(default_factory=lambda: CUA_HINH_MAT_TAP_TRUNG.mo_rong_bbox_mat)
    _tay: mp.solutions.hands.Hands = field(init=False, repr=False)
    _thoi_gian_bat_dau: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        self._tay = mp.solutions.hands.Hands(
            max_num_hands=self.so_tay_toi_da,
            min_detection_confidence=self.do_tin_cay_phat_hien,
            min_tracking_confidence=self.do_tin_cay_theo_doi
        )
    
    def analyze(self, khung_hinh: np.ndarray, khung_bbox_mat: Optional[dict] = None) -> dict:
        ket_qua = KetQuaTheoDoiTay()
        if khung_hinh is None:
            return ket_qua.thanh_dict()
            
        chieu_cao, chieu_rong = khung_hinh.shape[:2]
        anh_rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)
        ket_qua_mp = self._tay.process(anh_rgb)
        
        if not ket_qua_mp.multi_hand_landmarks:
            self._thoi_gian_bat_dau = None
            return ket_qua.thanh_dict()
        
        khung_bbox = None
        if khung_bbox_mat:
            khung_bbox = KhungBbox(khung_bbox_mat['x_min'], khung_bbox_mat['x_max'],
                                   khung_bbox_mat['y_min'], khung_bbox_mat['y_max'])
        
        ket_qua.so_tay_phat_hien = len(ket_qua_mp.multi_hand_landmarks)
        co_tay_gan = False
        
        for tay in ket_qua_mp.multi_hand_landmarks:
            ket_qua.diem_moc_tay.append(tay)
            xs = [lm.x * chieu_rong for lm in tay.landmark]
            ys = [lm.y * chieu_cao for lm in tay.landmark]
            ket_qua.khung_bbox_tay.append(KhungBbox(min(xs), max(xs), min(ys), max(ys)))
            
            if khung_bbox:
                # Center = giữa wrist và middle MCP
                tam = Diem2D(
                    (tay.landmark[0].x + tay.landmark[9].x) / 2,
                    (tay.landmark[0].y + tay.landmark[9].y) / 2
                )
                if khung_bbox.thua_chua(tam, self.mo_rong_bbox):
                    co_tay_gan = True
        
        ket_qua.tay_gan_mat = co_tay_gan
        
        # State machine: chỉ alert khi >threshold
        if co_tay_gan:
            if self._thoi_gian_bat_dau is None:
                self._thoi_gian_bat_dau = time.time()
            else:
                thoi_gian = time.time() - self._thoi_gian_bat_dau
                ket_qua.thoi_gian_mat_tap_trung = thoi_gian
                if thoi_gian >= self.nguong_thoi_gian:
                    ket_qua.canh_bao_mat_tap_trung = True
        else:
            self._thoi_gian_bat_dau = None
            
        return ket_qua.thanh_dict()
    
    def release(self) -> None:
        self._tay.close()
        
    def __enter__(self): return self
    def __exit__(self, *args): self.release()
