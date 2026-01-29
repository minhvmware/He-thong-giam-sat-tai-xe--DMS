#  Hệ Thống Giám Sát Tài Xế (DMS) - Documentation

##  Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt & Chạy](#cài-đặt--chạy)
4. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Thuật Toán](#thuật-toán)
8. [Ví Dụ Sử Dụng](#ví-dụ-sử-dụng)
9. [Troubleshooting](#troubleshooting)

---

##  Tổng Quan

**DMS (Driver Monitoring System)** là hệ thống thời gian thực phát hiện:
-  **Buồn ngủ** - Theo dõi nhắm mắt liên tục
-  **Ngáp** - Phát hiện miệng mở lớn
-  **Tư thế đầu** - Hướng nhìn không đúng
-  **Mất tập trung** - Tay gần mặt (dùng điện thoại)

**Công nghệ:**
- OpenCV: Xử lý hình ảnh
- MediaPipe: Phát hiện khuôn mặt/tay
- NumPy: Tính toán số học
- One-Euro Filter: Làm mịn tín hiệu

---

##  Yêu Cầu Hệ Thống

### Phần Cứng
```
Processor: Intel i5+ hoặc equivalent
RAM: 4GB minimum (8GB recommended)
Webcam: USB camera hoặc built-in
GPU: Không bắt buộc (sử dụng CPU)
```

### Phần Mềm
```
Python: 3.9+
OS: Windows 10+, Ubuntu 18+, macOS 10.14+
```

### Dependencies
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
playsound>=1.2.2 (tùy chọn, cho cảnh báo âm thanh)
```

---

##  Cài Đặt & Chạy

### 1. Clone/Download Dự Án
```bash
cd c:\Users\Quang Minh\OneDrive\PROJECT-22
```

### 2. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy Ứng Dụng
```bash
# Với camera mặc định (ID = 0)
python main.py

# Với camera khác (ID = 1)
python main.py --camera 1

# Với resolution khác
python main.py --width 1280 --height 720

# Tất cả tùy chọn
python main.py --camera 0 --width 640 --height 480
```

### 4. Dừng Ứng Dụng
Nhấn phím `q` trên cửa sổ video

---

##  Kiến Trúc Hệ Thống

### Sơ Đồ Flow

```
Input (Camera)
     ↓
┌─────────────────────────────┐
│  TienXuLyCLAHE              │ ← Tăng sáng ảnh
│  (Preprocessing)            │
└─────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│         Phân Tích Song Song         │
├─────────────────────────────────────┤
│ PhanTichMat           TheoDoiTay    │
│ (Face Analysis)       (Hand Track)  │
│ ├─ EAR (buồn ngủ)     ├─ Phát hiện │
│ ├─ MAR (ngáp)         └─ Vị trí    │
│ └─ Tư thế đầu                      │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────┐
│  BoLocOneEuro/              │ ← Làm mịn
│  BoLocOneEuroNhieuKenh      │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│  TraoDuaTinhNang            │ ← Hiển thị
│  (Visualization)            │   + Cảnh báo
└─────────────────────────────┘
     ↓
Output (Screen) + Alert Sound
```

### Module Structure

```
dms/
├── constants.py          # Cấu hình + ngưỡng
├── preprocessing.py      # CLAHE (tăng sáng)
├── filters.py            # One-Euro filter
├── face_analysis.py      # EAR, MAR, Head Pose
├── hand_tracking.py      # Phát hiện tay
├── visualization.py      # Vẽ + hiển thị
└── __init__.py          # Khởi tạo package

main.py                   # Điểm chính
```

---

## 🔌 API Reference

### PhanTichMat (Face Analysis)

```python
from dms.face_analysis import PhanTichMat

# Khởi tạo
analyzer = PhanTichMat(
    so_mat_toi_da=1,              # Max faces to detect
    do_tin_cay_phat_hien=0.5,     # Detection confidence
    do_tin_cay_theo_doi=0.5,      # Tracking confidence
    tinh_toan_diem_chi_tiet=True  # Refine landmarks
)

# Phân tích
result = analyzer.analyze(frame, timestamp=None)

# Output
{
    'mat_phat_hien': bool,           # Mặt được phát hiện?
    'ear': float,                     # Eye Aspect Ratio
    'mar': float,                     # Mouth Aspect Ratio
    'pitch': float,                   # Góc pitch (độ)
    'yaw': float,                     # Góc yaw (độ)
    'roll': float,                    # Góc roll (độ)
    'canh_bao_buon_ngu': bool,       # Buồn ngủ?
    'canh_bao_ngap': bool,            # Ngáp?
    'canh_bao_tu_the': bool,         # Tư thế sai?
    'diem_moc': list,                 # 468 landmarks
    'diem_moc_mat': dict,             # {'phai': [...], 'trai': [...]}
    'diem_moc_mieng': list,           # Mouth landmarks
    'vec_quay': ndarray,              # Rotation vector
    'vec_tuan': ndarray,              # Translation vector
    'khung_bbox_mat': dict            # {'x_min', 'x_max', 'y_min', 'y_max'}
}

# Giải phóng
analyzer.release()
```

### TheoDoiTay (Hand Tracking)

```python
from dms.hand_tracking import TheoDoiTay

# Khởi tạo
tracker = TheoDoiTay(
    so_tay_toi_da=2,                # Max hands
    do_tin_cay_phat_hien=0.5,
    do_tin_cay_theo_doi=0.5
)

# Phân tích
result = tracker.analyze(frame, khung_bbox_mat)

# Output
{
    'hands_detected': int,           # Số tay phát hiện
    'hand_landmarks': list,          # Landmarks mỗi tay
    'hand_bboxes': list,             # Bounding box mỗi tay
    'hand_near_face': bool,          # Tay gần mặt?
    'distraction_alert': bool,       # Mất tập trung?
    'distraction_duration': float    # Thời gian (giây)
}

tracker.release()
```

### TienXuLyCLAHE (Preprocessing)

```python
from dms.preprocessing import TienXuLyCLAHE

# Khởi tạo
preprocessor = TienXuLyCLAHE(
    han_clip=2.0,              # Clip limit
    kich_thuoc_o=(8, 8),       # Tile grid size
    khong_gian_mau='YCRCB'     # 'YCRCB' hoặc 'LAB'
)

# Xử lý
enhanced_frame = preprocessor.tang_cuong(frame)
```

### BoLocOneEuro (Filter)

```python
from dms.filters import BoLocOneEuro

# Khởi tạo
filter = BoLocOneEuro(
    cutoff_toi_thieu=1.0,  # Min cutoff
    beta=0.007,             # Movement sensitivity
    cutoff_dao_ham=1.0     # Derivative cutoff
)

# Lọc giá trị
smoothed = filter.loc(raw_value, timestamp)
```

---

## ⚙️ Configuration

### Sửa Đổi Ngưỡng

Chỉnh sửa `dms/constants.py`:

```python
@dataclass(frozen=True, slots=True)
class DrowsinessConfig:
    nguong_ear: float = 0.2      # ← Giảm = nhạy hơn
    so_khung_ear: int = 15       # ← Tăng = cần buồn ngủ lâu hơn
    nguong_mar: float = 1.3
    thoi_gian_canh_bao_am_thanh: float = 5.0  # ← Thời gian trước cảnh báo
    khoang_cach_am_thanh: float = 2.0         # ← Cooldown giữa cảnh báo
```

### Sửa Đổi File Âm Thanh

Trong `main.py`:

```python
@dataclass
class HeThongGiamSatTaiXe:
    duong_dan_am_thanh: str = "chicken-on-tree.mp3"  # ← Đổi đường dẫn
```

---

## 🧠 Thuật Toán

### 1. Eye Aspect Ratio (EAR)

**Công thức:**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

p1-p6: 6 landmarks mắt
```

**Ý nghĩa:**
- EAR ≈ 0.2 → Mắt nhắm
- EAR > 0.2 → Mắt mở

**Tham khảo:** Tereza et al., 2016

### 2. Mouth Aspect Ratio (MAR)

**Công thức:**
```
MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
```

**Ý nghĩa:**
- MAR > 1.3 → Ngáp

### 3. Head Pose Estimation

**Phương pháp:** solvePnP (Perspective-n-Point)

```
Input:
  - 3D model points (6 điểm mặt chuẩn)
  - 2D image points (6 landmarks trên ảnh)
  - Camera matrix K

Output:
  - Rotation vector (rvec) → Euler angles (pitch, yaw, roll)
  - Translation vector (tvec) → Vị trí đầu
```

**Thresholds:**
- Pitch > 20° → Nhìn xuống/lên
- Yaw > 30° → Nhìn sang
- Roll > 25° → Nghiêng đầu

### 4. One-Euro Filter

**Adaptive smoothing:**
```
fc(t) = f_min + β * |dx(t)|

α(f) = 1 / (1 + τ / te)
```

**Lợi ích:**
- Chậm → Mịn mạnh
- Nhanh → Mịn nhẹ
- Adaptive → Cân bằng giữa lag và noise

**Tham khảo:** Casiez et al., CHI 2012

### 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Tác dụng:**
- Xử lý cục bộ (local) thay vì toàn cục
- Giới hạn contrast → Tránh artifacts
- Tốt cho ảnh ánh sáng yếu

---

## 💡 Ví Dụ Sử Dụng

### Ví Dụ 1: Sử Dụng Trực Tiếp

```python
import cv2
from dms.face_analysis import PhanTichMat
from dms.preprocessing import TienXuLyCLAHE

# Khởi tạo
preprocessor = TienXuLyCLAHE()
analyzer = PhanTichMat()

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Xử lý
    enhanced = preprocessor.tang_cuong(frame)
    result = analyzer.analyze(enhanced)
    
    # Kiểm tra kết quả
    if result['canh_bao_buon_ngu']:
        print(" Phát hiện buồn ngủ!")
        print(f"EAR: {result['ear']:.2f}")
    
    if result['canh_bao_ngap']:
        print(" Phát hiện ngáp!")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

analyzer.release()
cap.release()
cv2.destroyAllWindows()
```

### Ví Dụ 2: Xử Lý File Video

```python
import cv2
import time
from dms.face_analysis import PhanTichMat

analyzer = PhanTichMat()
cap = cv2.VideoCapture('video.mp4')

# Lưu kết quả
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    ts = time.time()
    result = analyzer.analyze(frame, ts)
    
    results.append({
        'timestamp': ts,
        'ear': result['ear'],
        'drowsiness': result['canh_bao_buon_ngu']
    })

# Phân tích
drowsy_frames = [r for r in results if r['drowsiness']]
drowsy_percent = len(drowsy_frames) / len(results) * 100
print(f"Buồn ngủ: {drowsy_percent:.1f}%")

analyzer.release()
cap.release()
```

### Ví Dụ 3: Tùy Chỉnh Config

```python
import cv2
from dms.preprocessing import TienXuLyCLAHE
from dms.preprocessing import KhongGianMau

# CLAHE với LAB color space (chính xác hơn)
preprocessor = TienXuLyCLAHE(
    han_clip=4.0,           # Tăng độ tương phản
    khong_gian_mau=KhongGianMau.LAB
)

# Xử lý
frame = cv2.imread('image.jpg')
enhanced = preprocessor.tang_cuong(frame)
```

---

## 🔧 Troubleshooting

### Lỗi: "Không thể mở camera"

**Nguyên nhân:**
- Camera bị chiếm dụng bởi ứng dụng khác
- Driver camera không được cài đặt
- ID camera sai

**Giải pháp:**
```bash
# Thử camera ID khác
python main.py --camera 1

# Kiểm tra camera available (Windows)
ffmpeg -list_devices true -f dshow -i dummy

# Ubuntu
ls /dev/video*
```

### Lỗi: "playsound not installed"

**Giải pháp:**
```bash
pip install playsound
```

### Performance chậm

**Nguyên nhân:**
- CPU yếu
- Resolution quá cao
- Nhiều background processes

**Giải pháp:**
```bash
# Giảm resolution
python main.py --width 320 --height 240

# Hoặc tăng khoảng cách CLAHE tile
# Chỉnh sửa dms/constants.py:
# kich_thuoc_o: Tuple[int, int] = (16, 16)  # Tăng từ 8, 8
```

### Phát hiện không chính xác

**Nguyên nhân:**
- Ánh sáng yếu
- Góc camera xấu
- Khuôn mặt bị che phủ

**Giải pháp:**
```
1. Chỉnh sáng (tối nhất 50 lux)
2. Camera phía trước, ngang mặt
3. Làm sạch lens
4. Điều chỉnh ngưỡng trong constants.py
```

### Cảnh báo âm thanh không phát

**Kiểm tra:**
```python
# Test phát âm thanh
import playsound
playsound.playsound('chicken-on-tree.mp3')

# Kiểm tra volume hệ thống
# Windows: Volume mixer
# Linux: alsamixer
```

---

## Performance Metrics

### Yêu cầu

| Khía cạnh | Yêu cầu |
|----------|---------|
| Frame Rate | 25-30 FPS |
| Latency | <100ms |
| Memory | <500MB |
| CPU | 40-60% (i5+) |

### Optimization Tips

```python
# 1. Giảm resolution
python main.py --width 320 --height 240

# 2. Skip frames
if frame_count % 2 == 0:  # Process every 2nd frame
    result = analyzer.analyze(frame)

# 3. Async processing
import threading
thread = threading.Thread(target=analyzer.analyze, args=(frame,))
thread.start()
```

---

## Tham Khảo Khoa Học

1. **EAR/MAR:**
   - Tereza et al., "Real Time Eye Gaze Tracking with 3D Deformable Eye-Face Model", ICCV 2015

2. **Head Pose:**
   - Lepetit & Fua, "Keypoint Recognition using Randomized Trees", TPAMI 2006

3. **One-Euro Filter:**
   - Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012

4. **CLAHE:**
   - Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994

---

## Hỗ Trợ

Gặp vấn đề?

1. Kiểm tra **Troubleshooting** section
2. Xem **log messages** (terminal)
3. Thử giảm resolution hoặc điều chỉnh thresholds
4. Liên hệ: minhdev@duck.com

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-29  
**License:** MIT
