# Hệ Thống Giám Sát Tài Xế (DMS)

Hệ thống giám sát tài xế thời gian thực sử dụng OpenCV và MediaPipe, tối ưu hóa cho CPU.

## Tính Năng

- **Cải Thiện Ánh Sáng Yếu**: Tiền xử lý CLAHE để tăng khả năng hiển thị
- **Phát Hiện Buồn Ngủ**: Theo dõi Tỷ lệ Khung Mắt (EAR)
  - 🔔 **Cảnh báo âm thanh**: Phát âm thanh "chicken-on-tree.mp3" khi buồn ngủ liên tục >5 giây
- **Phát Hiện Ngáp**: Phân tích Tỷ lệ Khung Miệng (MAR)
- **Ước Lượng Tư Thế Đầu**: Phát hiện pitch/yaw/roll dựa trên PnP
- **Phát Hiện Mất Tập Trung**: Theo dõi tay đưa lên mặt
- **Ổn Định Tín Hiệu**: Bộ lọc One-Euro để giảm rung

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Cách Sử Dụng

```bash
python main.py
```

Nhấn `q` để thoát.

## Ngưỡng (có thể cấu hình trong `dms/constants.py`)

| Tham Số | Giá Trị | Mô Tả |
|---------|---------|-------|
| EAR_THRESHOLD | 0.20 | Ngưỡng nhắm mắt |
| EAR_CONSEC_FRAMES | 15 | Số khung hình cho cảnh báo buồn ngủ |
| MAR_THRESHOLD | 1.3 | Ngưỡng phát hiện ngáp |
| HEAD_POSE_PITCH | 20° | Góc pitch tối đa |
| HEAD_POSE_YAW | 30° | Góc yaw tối đa |
| DISTRACTION_TIME | 3.0s | Thời gian tay gần mặt |
| **DROWSINESS_ALERT_TIME** | **5.0s** | **Thời gian buồn ngủ trước khi phát âm thanh** |
| **ALERT_COOLDOWN** | **2.0s** | **Khoảng cách tối thiểu giữa 2 lần phát âm thanh** |

## Cấu Trúc Dự Án

```
PROJECT-22/
├── dms/
│   ├── __init__.py       # Khởi tạo gói
│   ├── constants.py      # Cấu hình
│   ├── preprocessing.py  # Cải thiện CLAHE
│   ├── filters.py        # Bộ lọc One-Euro
│   ├── face_analysis.py  # EAR, MAR, Tư thế đầu
│   ├── hand_tracking.py  # Phát hiện mất tập trung
│   └── visualization.py  # Lớp phủ trực quan
├── main.py               # Điểm khởi chạy
├── requirements.txt
└── README.md
```

## Tác Giả

Phát triển bởi MinhVM

## Giấy Phép

Dự án này được phát hành dưới giấy phép MIT.
