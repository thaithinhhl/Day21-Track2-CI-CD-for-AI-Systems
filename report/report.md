# Báo cáo Thực hành MLOps - Bước 3

## 3.6 So Sánh Kết Quả

Dưới đây là bảng so sánh hiệu suất của mô hình trước và sau khi thực hiện Huấn luyện liên tục (Continuous Training) với dữ liệu mới:

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | 0.6780 | 0.7540 |
| f1_score | 0.6767 | 0.7531 |

**Nhận xét:**
- Ở Bước 2, mô hình được huấn luyện trên tập dữ liệu ban đầu (2998 mẫu) cho ra độ chính xác (accuracy) là khoảng 67.8%.
- Ở Bước 3, sau khi hệ thống tự động nhận diện tập dữ liệu được bổ sung (lên mức 5996 mẫu) thông qua luồng CI/CD, độ chính xác đã tăng lên đáng kể đạt mức 75.4%.