# 0002 — Công thức Hybrid Centrality

- **Trạng thái:** Chưa quyết định
- **Ngày tạo:** 2026-09-04
- **Chặn:** Bước 9 (đóng góp nghiên cứu chính)
- **Cấu hình liên quan:** `configs/hybrid_centrality.yaml`

## Bối cảnh

Hybrid Centrality là **đóng góp chính** của đề tài, nhưng
`docs/XAI-Football_Project_Content.md` mục 10 ghi rõ: công thức toán học
chưa được đặc tả đầy đủ trong báo cáo tiến độ và **phải được định nghĩa
chính thức** trước khi triển khai.

`docs/Quy_trinh_XAI_Football.md` mục 12 cảnh báo thêm: hệ số `alpha` không
được chọn tùy ý, phải trình bày rõ cách xác định để hội đồng không đặt câu
hỏi về tính "thủ công, thiếu căn cứ".

## Dạng công thức đang đề xuất

```
HybridCentrality = alpha * Centrality_norm + (1 - alpha) * XAIImportance_norm
```

## Những điểm phải đặc tả

- [ ] **Dùng những độ đo centrality nào?** Degree (in/out), betweenness,
      closeness, eigenvector — dùng tất cả hay chọn lọc? Nếu dùng nhiều,
      gộp chúng lại bằng cách nào trước khi vào công thức?
- [ ] **Tổng hợp điểm XAI ra sao?** Trung bình hay trung bình có trọng số
      qua tất cả possession cầu thủ tham gia (mục 9.2)? Trọng số theo gì?
- [ ] **Chuẩn hóa bằng gì?** Min-max hay z-score? (Bắt buộc chuẩn hóa để
      thành phần có phương sai lớn không lấn át thành phần còn lại — mục 9.3)
- [ ] **Ngưỡng cỡ mẫu tối thiểu?** Cầu thủ tham gia quá ít possession có nên
      được xếp hạng không?
- [ ] **Đơn vị tổng hợp là gì?** Theo trận hay theo mùa giải?
- [ ] **Xác định alpha thế nào?** Dò trên tập validation theo tiêu chí nào?
      (Ví dụ: tối đa hóa tương quan Spearman với xA/xT — nhưng cần lập luận
      vì sao tiêu chí đó hợp lệ, tránh vòng lặp logic)
- [ ] **Lấy điểm XAI từ explainer nào?** GNNExplainer, PGExplainer, hay kết
      hợp? Phải chốt SAU khi có kết quả đối sánh ở Bước 8.

## Cách kiểm chứng đã dự kiến (mục 9.4)

1. So với centrality truyền thống thuần túy — baseline Tầng 3(b).
2. Đối chiếu định lượng với key passes, assists, xA, xT (face validity).
3. Đối chiếu định tính với vai trò thực tế của một số cầu thủ nổi bật.

## Quyết định

_Chưa chốt. Không được giả định công thức này đã hợp lệ trước khi kiểm chứng
thực nghiệm (`docs/XAI-Football_Project_Content.md` mục 18)._
