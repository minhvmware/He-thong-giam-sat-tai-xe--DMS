"""
Bộ lọc One-Euro - Làm mịn tín hiệu thời gian thực

Tự động điều chỉnh độ mịn: chậm→mịn mạnh, nhanh→mịn nhẹ.
Reference: Casiez et al., CHI 2012
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List
from .constants import CUA_HINH_BO_LOC


@dataclass
class BoLocThapThong:
    """Exponential smoothing: y = α*x + (1-α)*y_prev"""
    alpha: float = 1.0
    _y: Optional[float] = field(default=None, repr=False)
    
    def dat_lai(self) -> None:
        self._y = None
        
    def loc(self, x: float, alpha: Optional[float] = None) -> float:
        a = alpha if alpha is not None else self.alpha
        if self._y is None:
            self._y = x
        else:
            self._y = a * x + (1 - a) * self._y
        return self._y


@dataclass
class BoLocOneEuro:
    """
    Adaptive noise filter.
    - cutoff_toi_thieu: Độ mịn tối thiểu (thấp = mịn hơn)
    - beta: Độ nhạy với chuyển động nhanh
    """
    cutoff_toi_thieu: float = field(default_factory=lambda: CUA_HINH_BO_LOC.cutoff_toi_thieu)
    beta: float = field(default_factory=lambda: CUA_HINH_BO_LOC.beta)
    cutoff_dao_ham: float = field(default_factory=lambda: CUA_HINH_BO_LOC.cutoff_dao_ham)
    
    _bo_loc_x: BoLocThapThong = field(default_factory=BoLocThapThong, repr=False)
    _bo_loc_dx: BoLocThapThong = field(default_factory=BoLocThapThong, repr=False)
    _thoi_gian_truoc: Optional[float] = field(default=None, repr=False)
    
    def dat_lai(self) -> None:
        self._bo_loc_x.dat_lai()
        self._bo_loc_dx.dat_lai()
        self._thoi_gian_truoc = None
        
    def _tinh_alpha(self, cutoff: float, te: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def loc(self, x: float, timestamp: Optional[float] = None) -> float:
        timestamp = timestamp or time.time()
        
        if self._thoi_gian_truoc is None:
            self._thoi_gian_truoc = timestamp
            self._bo_loc_x._y = x
            self._bo_loc_dx._y = 0.0
            return x
            
        te = max(timestamp - self._thoi_gian_truoc, 1.0 / 30.0)
        self._thoi_gian_truoc = timestamp
        
        # Tính derivative
        dx = (x - (self._bo_loc_x._y or x)) / te
        alpha_d = self._tinh_alpha(self.cutoff_dao_ham, te)
        dx_lam_muot = self._bo_loc_dx.loc(dx, alpha_d)
        
        # Adaptive cutoff: fc = cutoff_toi_thieu + β*|dx|
        cutoff = self.cutoff_toi_thieu + self.beta * abs(dx_lam_muot)
        alpha = self._tinh_alpha(cutoff, te)
        
        return self._bo_loc_x.loc(x, alpha)


@dataclass
class BoLocOneEuroNhieuKenh:
    """One-Euro filter cho vector (VD: [pitch, yaw, roll])"""
    so_kenh: int = 3
    cutoff_toi_thieu: float = field(default_factory=lambda: CUA_HINH_BO_LOC.cutoff_toi_thieu)
    beta: float = field(default_factory=lambda: CUA_HINH_BO_LOC.beta)
    cutoff_dao_ham: float = field(default_factory=lambda: CUA_HINH_BO_LOC.cutoff_dao_ham)
    _danh_sach_bo_loc: List[BoLocOneEuro] = field(default_factory=list, repr=False)
    
    def __post_init__(self) -> None:
        self._danh_sach_bo_loc = [
            BoLocOneEuro(self.cutoff_toi_thieu, self.beta, self.cutoff_dao_ham)
            for _ in range(self.so_kenh)
        ]
        
    def dat_lai(self) -> None:
        for bo_loc in self._danh_sach_bo_loc:
            bo_loc.dat_lai()
            
    def loc(self, gia_tri: List[float], timestamp: Optional[float] = None) -> List[float]:
        return [self._danh_sach_bo_loc[i].loc(gia_tri[i], timestamp) for i in range(len(gia_tri))]
