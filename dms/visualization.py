"""
Visualization - Hiển thị kết quả DMS
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
from .constants import Mau, AlertType, EAR_THRESHOLD, MAR_THRESHOLD


@dataclass(frozen=True)
class CauHinhBangTin:
    x: int = 10
    y: int = 10
    chieu_rong: int = 190
    chieu_cao: int = 130


@dataclass
class TraoDuaTinhNang:
    bang_tin: CauHinhBangTin = field(default_factory=CauHinhBangTin)
    do_dai_truc: int = 100
    _mp_drawing: object = field(default=None, init=False, repr=False)
    _mp_hands: object = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        try:
            import mediapipe as mp
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
        except ImportError:
            pass
    
    def ve_luoi_mat(self, khung_hinh: np.ndarray, diem_moc) -> np.ndarray:
        if diem_moc is None:
            return khung_hinh
        chieu_cao, chieu_rong = khung_hinh.shape[:2]
        # Eye + lip landmarks
        for chi_so in (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
                    61, 39, 0, 269, 291, 405, 17, 181):
            cv2.circle(khung_hinh, (int(diem_moc[chi_so].x*chieu_rong), int(diem_moc[chi_so].y*chieu_cao)), 
                      1, Mau.XANH_LA, -1)
        return khung_hinh
    
    def ve_truc_tu_the_dau(self, khung_hinh: np.ndarray, vec_quay, vec_tuan, 
                            dau_mui: Tuple[float, float]) -> np.ndarray:
        if vec_quay is None:
            return khung_hinh
        chieu_cao, chieu_rong = khung_hinh.shape[:2]
        ma_tran_camera = np.array([[chieu_rong, 0, chieu_rong/2], [0, chieu_rong, chieu_cao/2], [0, 0, 1]], dtype=np.float64)
        truc = np.float64([[self.do_dai_truc, 0, 0], [0, self.do_dai_truc, 0], 
                           [0, 0, self.do_dai_truc]])
        diem_chieu, _ = cv2.projectPoints(truc, vec_quay, vec_tuan, ma_tran_camera, np.zeros((4, 1)))
        mui = (int(dau_mui[0]), int(dau_mui[1]))
        for i, mau_sac in enumerate([Mau.TRUC_X, Mau.TRUC_Y, Mau.TRUC_Z]):
            cv2.line(khung_hinh, mui, tuple(diem_chieu[i].ravel().astype(int)), mau_sac, 2)
        return khung_hinh
    
    def ve_so_lieu(self, khung_hinh: np.ndarray, ear: float, mar: float,
                     pitch: float, yaw: float, roll: float, fps: float) -> np.ndarray:
        p = self.bang_tin
        cv2.rectangle(khung_hinh, (p.x, p.y), (p.x+p.chieu_rong, p.y+p.chieu_cao), (0,0,0), -1)
        cv2.rectangle(khung_hinh, (p.x, p.y), (p.x+p.chieu_rong, p.y+p.chieu_cao), Mau.TRANG, 1)
        
        danh_sach_so_lieu = [
            (f"EAR: {ear:.2f}", Mau.DO if ear < EAR_THRESHOLD else Mau.XANH_LA),
            (f"MAR: {mar:.2f}", Mau.DO if mar > MAR_THRESHOLD else Mau.XANH_LA),
            (f"Pitch: {pitch:.1f}", Mau.TRANG),
            (f"Yaw: {yaw:.1f}", Mau.TRANG),
            (f"Roll: {roll:.1f}", Mau.TRANG),
            (f"FPS: {fps:.1f}", Mau.VANG),
        ]
        for i, (txt, mau_sac) in enumerate(danh_sach_so_lieu):
            cv2.putText(khung_hinh, txt, (p.x+10, p.y+25+i*18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, mau_sac, 1)
        return khung_hinh
    
    def ve_canh_bao(self, khung_hinh: np.ndarray, buon_ngu=False, ngap=False,
                    tu_the=False, mat_tap_trung=False) -> np.ndarray:
        chieu_cao, chieu_rong = khung_hinh.shape[:2]
        danh_sach_canh_bao = []
        if buon_ngu: danh_sach_canh_bao.append(AlertType.DROWSINESS.value)
        if ngap: danh_sach_canh_bao.append(AlertType.YAWN.value)
        if tu_the: danh_sach_canh_bao.append(AlertType.HEAD_POSE.value)
        if mat_tap_trung: danh_sach_canh_bao.append(AlertType.DISTRACTION.value)
        
        for i, txt in enumerate(danh_sach_canh_bao):
            y = chieu_cao - 30 - i*35
            sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x = (chieu_rong - sz[0]) // 2
            cv2.rectangle(khung_hinh, (x-10, y-25), (x+sz[0]+10, y+5), (0,0,0), -1)
            cv2.putText(khung_hinh, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Mau.DO, 2)
        return khung_hinh
    
    def ve_diem_moc_tay(self, khung_hinh: np.ndarray, danh_sach_tay: List) -> np.ndarray:
        if not danh_sach_tay or not self._mp_drawing:
            return khung_hinh
        for tay in danh_sach_tay:
            self._mp_drawing.draw_landmarks(
                khung_hinh, tay, self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing.DrawingSpec(color=Mau.XANH_LA, thickness=1),
                self._mp_drawing.DrawingSpec(color=Mau.TRANG, thickness=1))
        return khung_hinh
