# 🔌 API Reference - DMS

## Mục Lục

- [Constants](#constants)
- [Preprocessing](#preprocessing)
- [Face Analysis](#face-analysis)
- [Hand Tracking](#hand-tracking)
- [Filters](#filters)
- [Visualization](#visualization)

---

## Constants

### Module: `dms/constants.py`

#### 🎛️ DrowsinessConfig (Cấu Hình Buồn Ngủ)

```python
@dataclass(frozen=True, slots=True)
class DrowsinessConfig:
    """Ngưỡng phát hiện buồn ngủ"""
    
    nguong_ear: float = 0.2
    """Eye Aspect Ratio threshold. Giá trị thấp hơn = nhạy hơn"""
    
    so_khung_ear: int = 15
    """Số frame liên tục EAR < ngưỡng mới cảnh báo. 15 frame ≈ 0.5s @ 30fps"""
    
    nguong_mar: float = 1.3
    """Mouth Aspect Ratio threshold. Phát hiện ngáp khi MAR > ngưỡng"""
    
    thoi_gian_canh_bao_am_thanh: float = 5.0
    """Phát âm thanh khi buồn ngủ liên tục > 5 giây"""
    
    khoang_cach_am_thanh: float = 2.0
    """Cooldown tối thiểu giữa 2 lần phát âm thanh"""
```

#### 🎛️ HeadPoseConfig

```python
@dataclass(frozen=True, slots=True)
class HeadPoseConfig:
    """Ngưỡng tư thế đầu (độ)"""
    
    nguong_pitch: float = 20.0
    """Pitch > 20° = nhìn xuống/lên (bất thường)"""
    
    nguong_yaw: float = 30.0
    """Yaw > 30° = nhìn sang (không nhìn đường)"""
    
    nguong_roll: float = 25.0
    """Roll > 25° = nghiêng đầu"""
```

#### 🎛️ DistractionConfig

```python
@dataclass(frozen=True, slots=True)
class DistractionConfig:
    """Cấu hình phát hiện mất tập trung"""
    
    nguong_thoi_gian: float = 3.0
    """Thời gian tay gần mặt trước cảnh báo (giây)"""
    
    mo_rong_bbox_mat: float = 0.2
    """Mở rộng bbox mặt 20% khi kiểm tra tay gần mặt"""
```

#### 🎨 Mau (Màu BGR)

```python
class Mau:
    """Định nghĩa màu BGR cho visualization"""
    XANH_LA = (0, 255, 0)      # Green
    DO = (0, 0, 255)            # Red
    VANG = (0, 255, 255)        # Yellow
    TRANG = (255, 255, 255)    # White
    TRUC_X = (0, 0, 255)        # Red (X axis)
    TRUC_Y = (0, 255, 0)        # Green (Y axis)
    TRUC_Z = (255, 0, 0)        # Blue (Z axis)
```

---

## Preprocessing

### Module: `dms/preprocessing.py`

#### 📸 TienXuLyCLAHE

**Mục đích:** Cải thiện ảnh trong điều kiện ánh sáng yếu

```python
@dataclass
class TienXuLyCLAHE:
    """Tiền xử lý CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    
    han_clip: float = 2.0
    """Clip limit - Hạn chế khuếch đại contrast.
    Giá trị cao hơn = contrast lớn hơn (nhưng có noise)"""
    
    kich_thuoc_o: tuple = (8, 8)
    """Kích thước tile grid. (8,8) = 64 tiles.
    Giá trị nhỏ = chi tiết hơn, xử lý lâu hơn"""
    
    khong_gian_mau: KhongGianMau = KhongGianMau.YCRCB
    """YCRCB: Nhanh hơn (mặc định)
       LAB: Chính xác hơn về tri giác con người"""
```

**Method:**

```python
def tang_cuong(self, khung_hinh: np.ndarray) -> np.ndarray:
    """
    Tăng cường độ tương phản của ảnh
    
    Args:
        khung_hinh (np.ndarray): Ảnh input (BGR, uint8)
    
    Returns:
        np.ndarray: Ảnh đã xử lý (BGR, uint8)
    
    Example:
        >>> preprocessor = TienXuLyCLAHE()
        >>> enhanced = preprocessor.tang_cuong(frame)
    """
```

**Công thức CLAHE:**
```
1. Chuyển đổi RGB → YCrCb/LAB
2. Tách kênh Y (luminance)
3. Áp dụng CLAHE trên Y
4. Chuyển đổi ngược lại BGR
```

---

## Face Analysis

### Module: `dms/face_analysis.py`

#### 🎯 PhanTichMat

**Mục đích:** Phân tích khuôn mặt (EAR, MAR, Head Pose)

```python
@dataclass
class PhanTichMat:
    """Phân tích đặc trưng khuôn mặt"""
    
    so_mat_toi_da: int = 1
    """Số lượng mặt tối đa cần phát hiện"""
    
    do_tin_cay_phat_hien: float = 0.5
    """Detection confidence (0-1). Cao = chỉ phát hiện mặt rõ ràng"""
    
    do_tin_cay_theo_doi: float = 0.5
    """Tracking confidence (0-1). Cao = theo dõi ổn định"""
    
    tinh_toan_diem_chi_tiet: bool = True
    """Tính toán landmark chi tiết (468 vs 6 points)"""
```

**Method chính:**

```python
def analyze(self, khung_hinh: np.ndarray, timestamp: Optional[float] = None) -> dict:
    """
    Phân tích khuôn mặt trong frame
    
    Args:
        khung_hinh (np.ndarray): Frame video (BGR)
        timestamp (float, optional): Thời gian frame (Unix timestamp)
    
    Returns:
        dict: Kết quả phân tích
        {
            'mat_phat_hien': bool,
            'ear': float,                  # 0.0-1.0
            'mar': float,                  # 0.0-3.0
            'pitch': float,                # -90 to +90 độ
            'yaw': float,                  # -90 to +90 độ
            'roll': float,                 # -90 to +90 độ
            'canh_bao_buon_ngu': bool,
            'canh_bao_ngap': bool,
            'canh_bao_tu_the': bool,
            'diem_moc': list[468],         # MediaPipe landmarks
            'khung_bbox_mat': dict,        # x_min, x_max, y_min, y_max
            'vec_quay': ndarray (3,),      # Rotation vector
            'vec_tuan': ndarray (3,)       # Translation vector
        }
    
    Example:
        >>> analyzer = PhanTichMat()
        >>> result = analyzer.analyze(frame, time.time())
        >>> if result['canh_bao_buon_ngu']:
        ...     print("Phát hiện buồn ngủ!")
    """
```

**Method con:**

```python
def _tinh_ear(self, mat: List[np.ndarray]) -> float:
    """
    Tính Eye Aspect Ratio
    
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Args:
        mat (List): 6 landmark points của mắt
    
    Returns:
        float: EAR value (thường 0.1-0.5)
    """

def _tinh_mar(self, mieng: List[np.ndarray]) -> float:
    """
    Tính Mouth Aspect Ratio
    
    Formula: MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
    
    Args:
        mieng (List): 8 landmark points của miệng
    
    Returns:
        float: MAR value (thường 0.1-2.5)
    """

def _uoc_luong_tu_the(self, diem_moc, chieu_rong: int, chieu_cao: int) -> HeadPose:
    """
    Ước lượng tư thế đầu bằng solvePnP
    
    Input:
        - 6 3D model points (chuẩn)
        - 6 2D image points (detected)
        - Camera matrix (từ frame size)
    
    Output:
        - Euler angles (pitch, yaw, roll)
        - Rotation & translation vectors
    """

def release(self) -> None:
    """Giải phóng MediaPipe resources"""
```

#### 📐 HeadPose (NamedTuple)

```python
class HeadPose(NamedTuple):
    pitch: float                          # Góc pitch (độ)
    yaw: float                            # Góc yaw (độ)
    roll: float                           # Góc roll (độ)
    rvec: Optional[np.ndarray] = None    # Rotation vector (3,)
    tvec: Optional[np.ndarray] = None    # Translation vector (3,)
```

---

## Hand Tracking

### Module: `dms/hand_tracking.py`

#### ✋ TheoDoiTay

**Mục đích:** Phát hiện & theo dõi tay, phát hiện mất tập trung

```python
@dataclass
class TheoDoiTay:
    """Theo dõi tay & phát hiện mất tập trung"""
    
    so_tay_toi_da: int = 2
    """Số lượng tay tối đa phát hiện"""
    
    do_tin_cay_phat_hien: float = 0.5
    """Detection confidence"""
    
    do_tin_cay_theo_doi: float = 0.5
    """Tracking confidence"""
    
    nguong_thoi_gian: float = 3.0
    """Thời gian tay gần mặt trước cảnh báo (giây)"""
    
    mo_rong_bbox: float = 0.2
    """Mở rộng bbox mặt khi kiểm tra (20%)"""
```

**Method:**

```python
def analyze(self, khung_hinh: np.ndarray, khung_bbox_mat: Optional[dict] = None) -> dict:
    """
    Phân tích tay trong frame
    
    Args:
        khung_hinh (np.ndarray): Frame video
        khung_bbox_mat (dict): Face bounding box
        {
            'x_min': float (0-1),
            'x_max': float (0-1),
            'y_min': float (0-1),
            'y_max': float (0-1)
        }
    
    Returns:
        dict:
        {
            'hands_detected': int,           # Số tay phát hiện
            'hand_landmarks': list,          # Landmarks của mỗi tay
            'hand_bboxes': list,             # Bounding box mỗi tay
            'hand_near_face': bool,          # Tay gần mặt?
            'distraction_alert': bool,       # Cảnh báo mất tập trung?
            'distraction_duration': float    # Thời gian (giây)
        }
    """
```

**State Machine:**
```
No hand → None
    ↓
Hand near face detected
    ↓
< 3s: No alert
    ↓
>= 3s: ALERT!
    ↓
Hand leaves
    ↓
Reset
```

---

## Filters

### Module: `dms/filters.py`

#### 📊 BoLocThapThong (Low Pass Filter)

```python
@dataclass
class BoLocThapThong:
    """Bộ lọc thông thấp (Exponential smoothing)"""
    
    alpha: float = 1.0
    """Hệ số smoothing (0-1). 1.0 = không lọc, 0.0 = lọc mạnh"""
```

**Formula:**
```
y(t) = α * x(t) + (1 - α) * y(t-1)
```

#### 🎯 BoLocOneEuro

```python
@dataclass
class BoLocOneEuro:
    """One-Euro Filter - Adaptive noise filtering"""
    
    cutoff_toi_thieu: float = 1.0
    """Minimum cutoff frequency (Hz)"""
    
    beta: float = 0.007
    """Speed coefficient. Cao = nhạy với chuyển động nhanh"""
    
    cutoff_dao_ham: float = 1.0
    """Derivative cutoff (Hz)"""
```

**Method:**

```python
def loc(self, x: float, timestamp: Optional[float] = None) -> float:
    """
    Lọc giá trị đơn lẻ
    
    Args:
        x (float): Giá trị input
        timestamp (float): Unix timestamp
    
    Returns:
        float: Giá trị đã lọc
    
    Adaptive behavior:
        - Chuyển động slow → Lọc mạnh (smooth)
        - Chuyển động fast → Lọc nhẹ (responsive)
    """
```

#### 📊 BoLocOneEuroNhieuKenh

```python
@dataclass
class BoLocOneEuroNhieuKenh:
    """One-Euro Filter cho vector (pitch, yaw, roll)"""
    
    so_kenh: int = 3
    """Số channels (3 cho pitch/yaw/roll)"""
```

**Method:**

```python
def loc(self, gia_tri: List[float], timestamp: Optional[float] = None) -> List[float]:
    """
    Lọc vector giá trị
    
    Args:
        gia_tri: [pitch, yaw, roll] hoặc bất kỳ 3 giá trị
        timestamp: Unix timestamp
    
    Returns:
        List[float]: Vector đã lọc
    
    Example:
        >>> filter = BoLocOneEuroNhieuKenh(3)
        >>> smoothed = filter.loc([pitch, yaw, roll], time.time())
    """
```

---

## Visualization

### Module: `dms/visualization.py`

#### 🎨 TraoDuaTinhNang

```python
@dataclass
class TraoDuaTinhNang:
    """Hiển thị kết quả lên frame"""
    
    bang_tin: CauHinhBangTin = field(default_factory=CauHinhBangTin)
    """Cấu hình vị trí bảng thông tin"""
    
    do_dai_truc: int = 100
    """Độ dài trục 3D (pixels)"""
```

**Methods:**

```python
def ve_luoi_mat(self, khung_hinh: np.ndarray, diem_moc) -> np.ndarray:
    """
    Vẽ các landmark quan trọng trên mặt
    - 12 mắt + 8 miệng landmarks
    - Màu xanh lá
    """

def ve_truc_tu_the_dau(self, khung_hinh: np.ndarray, vec_quay, vec_tuan, 
                       dau_mui: Tuple[float, float]) -> np.ndarray:
    """
    Vẽ 3 trục tư thế đầu (X, Y, Z)
    - Đỏ: X axis
    - Xanh: Y axis
    - Xanh dương: Z axis
    """

def ve_so_lieu(self, khung_hinh: np.ndarray, ear: float, mar: float,
               pitch: float, yaw: float, roll: float, fps: float) -> np.ndarray:
    """
    Vẽ bảng thông tin:
    - EAR: (đỏ nếu < ngưỡng)
    - MAR: (đỏ nếu > ngưỡng)
    - Pitch, Yaw, Roll
    - FPS (vàng)
    """

def ve_canh_bao(self, khung_hinh: np.ndarray, buon_ngu=False, ngap=False,
                tu_the=False, mat_tap_trung=False) -> np.ndarray:
    """
    Vẽ cảnh báo ở dưới cùng frame
    - Text đỏ, nền đen
    - Căn giữa
    """

def ve_diem_moc_tay(self, khung_hinh: np.ndarray, danh_sach_tay: List) -> np.ndarray:
    """
    Vẽ skeleton tay
    - Xanh lá: joints
    - Trắng: connections
    """
```

#### 🎛️ CauHinhBangTin

```python
@dataclass(frozen=True)
class CauHinhBangTin:
    """Cấu hình vị trí bảng thông tin"""
    
    x: int = 10          # Pixel từ trái
    y: int = 10          # Pixel từ trên
    chieu_rong: int = 190
    chieu_cao: int = 130
```

---

## 🔗 Integrations

### Lớp Chính: HeThongGiamSatTaiXe (main.py)

```python
@dataclass
class HeThongGiamSatTaiXe:
    """Hệ thống chính - tích hợp tất cả"""
    
    cau_hinh_camera: CauHinhCamera
    """Cấu hình camera"""
    
    ten_cua_so: str = "He Thong Giam Sat Tai Xe"
    """Tên cửa sổ hiển thị"""
    
    duong_dan_am_thanh: str = "chicken-on-tree.mp3"
    """Đường dẫn file âm thanh cảnh báo"""

def chay(self) -> None:
    """Chạy hệ thống"""

def _xu_ly(self, khung_hinh: np.ndarray) -> np.ndarray:
    """
    Xử lý 1 frame:
    1. Tăng cường
    2. Phân tích mặt
    3. Theo dõi tay
    4. Cảnh báo âm thanh
    5. Hiển thị
    """

def _dung(self) -> None:
    """Dừng & giải phóng resources"""
```

---

## 📝 Ghi Chú

- Tất cả hàm **không** raise exceptions, trả về default nếu lỗi
- Sử dụng **logging** cho debug info
- Type hints **bắt buộc** cho tất cả hàm
- Docstring theo **Google style**

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-29
