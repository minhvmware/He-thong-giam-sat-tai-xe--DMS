"""
Face Analysis - EAR, MAR, Head Pose

- EAR (Eye Aspect Ratio): Phát hiện buồn ngủ
- MAR (Mouth Aspect Ratio): Phát hiện ngáp
- Head Pose: Ước lượng góc quay đầu bằng PnP
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, NamedTuple
import cv2
import numpy as np
import mediapipe as mp
from .constants import (
    CHI_SO_MAT_PHAI, CHI_SO_MAT_TRAI, CHI_SO_MIENG_NGOAI,
    CHI_SO_TU_THE, DIEM_MAT_3D,
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, MAR_THRESHOLD,
    HEAD_POSE_PITCH_THRESHOLD, HEAD_POSE_YAW_THRESHOLD
)
from .filters import BoLocOneEuro, BoLocOneEuroNhieuKenh


class HeadPose(NamedTuple):
    pitch: float
    yaw: float
    roll: float
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


@dataclass
class KetQuaPhanTichMat:
    mat_phat_hien: bool = False
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    canh_bao_buon_ngu: bool = False
    canh_bao_ngap: bool = False
    canh_bao_tu_the: bool = False
    diem_moc: Optional[list] = None
    diem_moc_mat: Optional[dict] = None
    diem_moc_mieng: Optional[list] = None
    vec_quay: Optional[np.ndarray] = None
    vec_tuan: Optional[np.ndarray] = None
    khung_bbox_mat: Optional[dict] = None


@dataclass
class PhanTichMat:
    so_mat_toi_da: int = 1
    do_tin_cay_phat_hien: float = 0.5
    do_tin_cay_theo_doi: float = 0.5
    tinh_toan_diem_chi_tiet: bool = True
    
    _luoi_mat: mp.solutions.face_mesh.FaceMesh = field(init=False, repr=False)
    _bo_loc_ear: BoLocOneEuro = field(default_factory=BoLocOneEuro, repr=False)
    _bo_loc_mar: BoLocOneEuro = field(default_factory=BoLocOneEuro, repr=False)
    _bo_loc_tu_the: BoLocOneEuroNhieuKenh = field(
        default_factory=lambda: BoLocOneEuroNhieuKenh(3), repr=False)
    _dem_ear: int = field(default=0, repr=False)
    _he_so_sai_le: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 1), dtype=np.float64), repr=False)
    
    def __post_init__(self) -> None:
        self._luoi_mat = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.so_mat_toi_da,
            refine_landmarks=self.tinh_toan_diem_chi_tiet,
            min_detection_confidence=self.do_tin_cay_phat_hien,
            min_tracking_confidence=self.do_tin_cay_theo_doi
        )
    
    def _tinh_ear(self, mat: List[np.ndarray]) -> float:
        """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
        if len(mat) != 6:
            return 0.0
        A = np.linalg.norm(mat[1] - mat[5])
        B = np.linalg.norm(mat[2] - mat[4])
        C = np.linalg.norm(mat[0] - mat[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    
    def _tinh_mar(self, mieng: List[np.ndarray]) -> float:
        """MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|)"""
        if len(mieng) != 8:
            return 0.0
        A = np.linalg.norm(mieng[1] - mieng[7])
        B = np.linalg.norm(mieng[2] - mieng[6])
        C = np.linalg.norm(mieng[3] - mieng[5])
        D = np.linalg.norm(mieng[0] - mieng[4])
        return (A + B + C) / (2.0 * D) if D > 0 else 0.0
    
    def _uoc_luong_tu_the(self, diem_moc, chieu_rong: int, chieu_cao: int) -> HeadPose:
        """Ước lượng pose bằng solvePnP."""
        ma_tran_camera = np.array([[chieu_rong, 0, chieu_rong/2], [0, chieu_rong, chieu_cao/2], [0, 0, 1]], dtype=np.float64)
        diem_chieu = np.array([[diem_moc[i].x*chieu_rong, diem_moc[i].y*chieu_cao] 
                        for i in CHI_SO_TU_THE], dtype=np.float64)
        
        ok, vec_quay, vec_tuan = cv2.solvePnP(
            DIEM_MAT_3D, diem_chieu, ma_tran_camera, self._he_so_sai_le)
        
        if not ok:
            return HeadPose(0, 0, 0)
        
        ma_tran_quay, _ = cv2.Rodrigues(vec_quay)
        sy = np.sqrt(ma_tran_quay[0,0]**2 + ma_tran_quay[1,0]**2)
        
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(-ma_tran_quay[2,0], sy))
            yaw = np.degrees(np.arctan2(ma_tran_quay[1,0], ma_tran_quay[0,0]))
            roll = np.degrees(np.arctan2(ma_tran_quay[2,1], ma_tran_quay[2,2]))
        else:
            pitch = np.degrees(np.arctan2(-ma_tran_quay[2,0], sy))
            yaw = 0
            roll = np.degrees(np.arctan2(-ma_tran_quay[1,2], ma_tran_quay[1,1]))
            
        return HeadPose(pitch, yaw, roll, vec_quay, vec_tuan)
    
    def _trich_xuat(self, diem_moc, chi_so, chieu_rong, chieu_cao) -> List[np.ndarray]:
        return [np.array([diem_moc[i].x*chieu_rong, diem_moc[i].y*chieu_cao]) for i in chi_so]
    
    def analyze(self, khung_hinh: np.ndarray, timestamp: Optional[float] = None) -> dict:
        ket_qua = KetQuaPhanTichMat()
        if khung_hinh is None or khung_hinh.size == 0:
            return self._thanh_dict(ket_qua)
            
        chieu_cao, chieu_rong = khung_hinh.shape[:2]
        anh_rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)
        ket_qua_luoi = self._luoi_mat.process(anh_rgb)
        
        if not ket_qua_luoi.multi_face_landmarks:
            self._dem_ear = 0
            return self._thanh_dict(ket_qua)
        
        diem_moc = ket_qua_luoi.multi_face_landmarks[0].landmark
        ket_qua.mat_phat_hien = True
        ket_qua.diem_moc = diem_moc
        
        # Bbox
        danh_sach_x, danh_sach_y = [l.x for l in diem_moc], [l.y for l in diem_moc]
        ket_qua.khung_bbox_mat = {'x_min': min(danh_sach_x), 'x_max': max(danh_sach_x), 
                       'y_min': min(danh_sach_y), 'y_max': max(danh_sach_y)}
        
        # EAR
        phai = self._trich_xuat(diem_moc, CHI_SO_MAT_PHAI, chieu_rong, chieu_cao)
        trai = self._trich_xuat(diem_moc, CHI_SO_MAT_TRAI, chieu_rong, chieu_cao)
        ket_qua.diem_moc_mat = {'phai': phai, 'trai': trai}
        ear_thom = (self._tinh_ear(phai) + self._tinh_ear(trai)) / 2
        ket_qua.ear = self._bo_loc_ear.loc(ear_thom, timestamp)
        
        if ket_qua.ear < EAR_THRESHOLD:
            self._dem_ear += 1
            ket_qua.canh_bao_buon_ngu = self._dem_ear >= EAR_CONSEC_FRAMES
        else:
            self._dem_ear = 0
        
        # MAR
        mieng = self._trich_xuat(diem_moc, CHI_SO_MIENG_NGOAI, chieu_rong, chieu_cao)
        ket_qua.diem_moc_mieng = mieng
        ket_qua.mar = self._bo_loc_mar.loc(self._tinh_mar(mieng), timestamp)
        ket_qua.canh_bao_ngap = ket_qua.mar > MAR_THRESHOLD
        
        # Head pose
        tu_the = self._uoc_luong_tu_the(diem_moc, chieu_rong, chieu_cao)
        lam_muot = self._bo_loc_tu_the.loc([tu_the.pitch, tu_the.yaw, tu_the.roll], timestamp)
        ket_qua.pitch, ket_qua.yaw, ket_qua.roll = lam_muot
        ket_qua.vec_quay = tu_the.rvec
        ket_qua.vec_tuan = tu_the.tvec
        ket_qua.canh_bao_tu_the = abs(ket_qua.pitch) > HEAD_POSE_PITCH_THRESHOLD or \
                            abs(ket_qua.yaw) > HEAD_POSE_YAW_THRESHOLD
        
        return self._thanh_dict(ket_qua)
    
    @staticmethod
    def _thanh_dict(ket_qua: KetQuaPhanTichMat) -> dict:
        return {k: getattr(ket_qua, k) for k in ket_qua.__dataclass_fields__}
    
    def release(self) -> None:
        self._luoi_mat.close()
        
    def __enter__(self): return self
    def __exit__(self, *args): self.release()
