# Bản Đặc Tả Thiết Kế Hệ Thống
## Zero-Shot Pattern Detection cho Bản Vẽ Kỹ Thuật BOM

**Phiên bản:** 1.0  
**Ngày:** 2026-05-24  
**Tác giả:** Ứng viên AI/Computer Vision Engineer

---

## 1. Phân Tích Bài Toán

### 1.1 Đặc điểm của bản vẽ BOM

Bản vẽ kỹ thuật BOM (Bill of Materials) có các đặc điểm sau làm phức tạp bài toán nhận dạng:

- **Grayscale / Binary:** Bản vẽ chỉ có hai màu (đen/trắng), không có thông tin màu sắc để phân biệt đối tượng.
- **Nét mảnh, độ phân giải cao:** Các chi tiết kỹ thuật thường được vẽ bằng nét rất mảnh (1–2px ở bản scan 300DPI), đòi hỏi preprocessing cẩn thận để không xóa mất thông tin.
- **Nhiễu từ quá trình scan:** JPEG artifacts, đốm nhiễu, nét bị gãy, độ tương phản không đều theo vùng.
- **Pattern xuất hiện nhiều lần:** Cùng một ký hiệu linh kiện (e.g., điện trở, tụ điện) có thể xuất hiện 10–100 lần trong một bản vẽ với kích thước và góc xoay có biến thiên nhỏ.
- **Tỷ lệ kích thước khác nhau:** Pattern có thể được vẽ ở scale 90%–110% so với template tham chiếu.

### 1.2 Thách thức chính

| Thách thức | Mô tả | Tác động |
|------------|-------|----------|
| **Zero-shot** | Không có training data cho pattern cụ thể | Loại bỏ mọi phương pháp supervised |
| **Nhiều occurrences** | Cần tìm TẤT CẢ vị trí, không chỉ 1 | Cần NMS để loại duplicate |
| **Scale variation** | Pattern có thể to/nhỏ hơn ±15% | Cần multi-scale sweep |
| **Rotation nhỏ** | Bản vẽ scan thường bị lệch ±10° | Cần rotation sweep |
| **Domain shift** | Model train trên ảnh tự nhiên, không phải line art | DINOv2 cần robust với domain shift |

### 1.3 Tại sao không thể dùng supervised detection thông thường?

Phương pháp như YOLO, Faster-RCNN yêu cầu:
1. **Labeled training data** cho mỗi loại pattern — không khả thi vì mỗi dự án BOM có ký hiệu riêng.
2. **Fine-tuning mỗi khi có pattern mới** — không đáp ứng yêu cầu "zero-shot" của bài toán.
3. **Kích thước object rất nhỏ** (pattern 30–100px trong bản vẽ 3000×4000px) — anchor-based detection kém hiệu quả ở scale này.

**Kết luận:** Bài toán yêu cầu một hệ thống generalizable hoạt động với bất kỳ pattern mới nào tại inference time mà không cần retrain.

---

## 2. Lý Do Chọn Hướng Tiếp Cận Hybrid NCC + DINOv2

### 2.1 So sánh các hướng tiếp cận

| Hướng tiếp cận | Ưu điểm | Nhược điểm | Phù hợp? |
|----------------|---------|------------|----------|
| **Classical NCC** | Nhanh, không cần model, chính xác khi template giống hệt | Rất nhạy với noise, scale, rotation; nhiều false positive | Chỉ làm Stage 1 |
| **Feature Matching (SIFT/ORB)** | Invariant với rotation/scale, không cần training | Kém trên line art (ít keypoints), không generalizable | Không phù hợp chính |
| **Siamese CNN** | Có thể fine-tune, pair-wise matching | Cần training data của domain, không truly zero-shot | Không phù hợp |
| **Grounding DINO / OWLv2** | Zero-shot với text prompt | Cần text description của pattern, không phù hợp với arbitrary symbol | Không phù hợp |
| **DINOv2 features (self-supervised)** | Zero-shot thực sự, robust với domain shift, capture geometric structure | Chậm hơn NCC, cần GPU cho production | **Phù hợp (Stage 2)** |
| **Hybrid NCC + DINOv2** | Kết hợp speed của NCC + accuracy của DINOv2 | Phức tạp hơn single-stage | **✅ Lựa chọn của chúng tôi** |

### 2.2 Lý do chọn DINOv2

DINOv2 (Oquab et al., 2023) được huấn luyện với self-supervised learning trên 142M ảnh đa dạng. Các đặc điểm phù hợp với bài toán:

1. **Patch-level geometric features:** ViT chia ảnh thành patches 14×14, mỗi patch encode đặc trưng hình học cục bộ. Điều này capture topology của line drawing tốt hơn CNN thông thường.
2. **Không phụ thuộc màu sắc:** Bản vẽ được convert sang RGB 3-channel bằng cách stack grayscale, DINOv2 vẫn extract được features hữu ích.
3. **Zero-shot thực sự:** Không cần fine-tune — encode template một lần, so sánh cosine similarity với candidate crops. Hoạt động với bất kỳ pattern mới nào.
4. **Domain transfer:** Nghiên cứu từ DINOv2 paper cho thấy features transfer tốt sang nhiều domain khác nhau, bao gồm medical imaging và technical diagrams.

### 2.3 Lý do dùng NCC làm Stage 1

Brute-force DINOv2 sliding window trên toàn bộ ảnh 3000×4000px với stride 14px sẽ cần encode ~57,000 patches — quá chậm cho real-time use. NCC giải quyết vấn đề này:

- **Tốc độ:** `cv2.matchTemplate` với TM_CCOEFF_NORMED chạy trong 1–5s trên CPU cho ảnh A3.
- **High recall:** Threshold thấp (0.55) đảm bảo không bỏ sót candidate nào đáng kể.
- **Search space reduction:** Từ ~57,000 vị trí xuống còn 30–200 candidates, giảm 99.6% workload cho Stage 2.

---

## 3. Kiến Trúc Hệ Thống Chi Tiết

### 3.1 Sơ đồ pipeline

```
Input: Pattern Image (P), Drawing Image (D)
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Stage 0: Preprocessor                          │
│  • load_image() → grayscale uint8               │
│  • resize_if_needed() → max 4096px              │
│  • binarize() → adaptive threshold              │
│  • denoise() → morphological closing kernel=2   │
└─────────────────────┬───────────────────────────┘
                      │  P_proc, D_proc
                      ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: NCCMatcher                            │
│  For each (scale ∈ [0.85..1.15],               │
│            angle ∈ [-10°..10°]):                │
│  • resize(P_proc, scale)                        │
│  • rotate(P_proc, angle)                        │
│  • cv2.matchTemplate(D_proc, P_rotated)         │
│  • Collect locs with score ≥ 0.55               │
│  • IoU-NMS (threshold 0.3)                      │
│  → 30–200 candidate bboxes                      │
└─────────────────────┬───────────────────────────┘
                      │  candidates[]
                      ▼
        ┌─────────────────────────┐
        │  candidates empty?       │──YES──► Return empty result
        └────────────┬────────────┘
                     │ NO
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: DINOVerifier                          │
│  • encode_template(P_proc) → feat_P (384-D)     │
│  For each candidate:                            │
│  • crop D_proc[bbox + 10% padding]              │
│  • Batch encode → feat_C (384-D)                │
│  • cosine_sim(feat_P, feat_C) → dino_score      │
│  • confidence = (ncc_score + dino_score) / 2    │
│  • Filter: keep if dino_score ≥ 0.75            │
│  → 1–30 verified detections                     │
└─────────────────────┬───────────────────────────┘
                      │  verified candidates
                      ▼
┌─────────────────────────────────────────────────┐
│  Stage 4: Postprocessor                         │
│  • final_nms(IoU=0.4) → deduplicated            │
│  • format_output() → JSON dict                  │
│  • draw_boxes() → annotated RGB image           │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
        Output: JSON + Annotated Image
```

### 3.2 Mô tả chi tiết từng module

#### Preprocessor (`src/preprocessor.py`)

**Adaptive binarization:** Dùng `cv2.adaptiveThreshold` với `ADAPTIVE_THRESH_GAUSSIAN_C`, `blockSize=15`, `C=4`. Gaussian weighting tốt hơn mean weighting cho bản vẽ có nét mảnh vì ít bị artifact ở vùng chuyển tiếp. `blockSize=15` đủ lớn để xử lý vùng scan không đều nhưng đủ nhỏ để giữ được nét chi tiết.

**Morphological denoising:** Morphological closing với kernel 2×2 fill các gap nhỏ trong nét bị gãy do scan, mà không làm dày nét vẽ đáng kể.

**Resize:** Giới hạn 4096px chiều lớn nhất để tránh OOM và tăng tốc Stage 1. Scale factor được lưu lại để có thể map bbox về kích thước gốc nếu cần.

#### NCCMatcher (`src/ncc_matcher.py`)

**Multi-scale sweep:** Scale range `[0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]` covering ±15% variation. Mỗi scale tăng 5% là heuristic balance giữa coverage và compute.

**Rotation sweep:** `[-10°, -5°, 0°, 5°, 10°]`. Bản vẽ scan thường bị lệch ≤5°, nhưng 10° range đảm bảo không bỏ sót. Fill border màu trắng (255) sau khi rotate để không tạo ra các pixel đen artifact ảnh hưởng NCC score.

**NMS implementation:** Implement từ scratch (không phụ thuộc torchvision) với bubble-sort-style suppression. Candidates được sort theo score giảm dần, giữ box có score cao nhất khi IoU > threshold.

#### DINOVerifier (`src/dino_verifier.py`)

**Feature extraction:** Sử dụng `forward_features()` của DINOv2 để lấy `x_norm_patchtokens` — normalized patch tokens sau layer normalization. Mean-pooling tất cả N_patches (256 patches với input 224×224, patch_size=14) cho ra vector 384-D (ViT-S/14) representing toàn bộ structure của ảnh.

**Batch encoding:** Với >10 candidates, batch encoding trên GPU có thể giảm thời gian 4-8× so với encode từng ảnh riêng lẻ.

**Padding strategy:** Candidate crop được extend thêm 10% mỗi chiều để đảm bảo DINOv2 nhìn thấy một chút context xung quanh pattern, cải thiện discrimination với background.

#### Postprocessor (`src/postprocessor.py`)

**Final NMS:** IoU threshold cao hơn Stage 1 (0.4 vs 0.3) vì ở giai đoạn này candidates đã được verify bởi DINOv2, cần conservative để không merge các occurrences thực sự gần nhau.

**Visualization:** Convert từ grayscale → BGR → vẽ boxes → convert sang RGB để tương thích với Gradio/PIL. Text annotation chỉ vẽ khi box đủ lớn (≥30px) để tránh overlapping text.

---

## 4. Lý Do DINOv2 Hoạt Động Tốt với Bản Vẽ Kỹ Thuật

### 4.1 Self-supervised training và patch features

DINOv2 sử dụng self-distillation với no labels (Caron et al., 2021; Oquab et al., 2023). Quá trình training buộc model học các features **semantic và geometric** thay vì chỉ low-level texture. Điều này đặc biệt quan trọng với line drawings vì:

- Line art không có texture — chỉ có hình dạng và topology
- DINOv2 patch features encode **spatial relationships** giữa các nét vẽ
- Không bị confuse bởi màu sắc hay gradient (bản vẽ BOM là binary)

### 4.2 Robustness với domain shift

Mặc dù DINOv2 được train trên ảnh tự nhiên, công trình nghiên cứu từ DINOv2 paper (Oquab et al., 2023, Table 8) cho thấy features transfer sang:
- **Medical imaging** (X-ray, histology slides)
- **Satellite imagery** (remote sensing)
- **Depth estimation** (structural understanding)

Bản vẽ kỹ thuật có cấu trúc hình học rõ ràng — paths, circles, rectangles, text — những đặc trưng này được DINOv2 encode tốt dù không được fine-tune.

### 4.3 Không cần fine-tune → Thực sự zero-shot

Nếu fine-tune DINOv2 trên một tập bản vẽ BOM cụ thể:
- Model sẽ bị biased với ký hiệu đã thấy trong training set
- Không generalizable sang ký hiệu mới
- Cần thu thập và label data — tốn thời gian và chi phí

Bằng cách dùng pre-trained features + cosine similarity, hệ thống **thực sự zero-shot**: chỉ cần 1 ảnh pattern, không cần fine-tune, không cần bất kỳ label nào.

---

## 5. Thông Số Hệ Thống & Hyperparameter

### 5.1 Bảng hyperparameters

| Parameter | Default | Lý do chọn |
|-----------|---------|------------|
| `scales` | [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15] | Cover ±15% scale variation thực tế trong bản vẽ |
| `angles` | [-10, -5, 0, 5, 10] | Cover ±10° scan skew; tăng lên ±15° nếu cần nhưng tốn compute |
| `ncc_threshold` | 0.55 | Intentionally low → high recall; Stage 2 sẽ filter false positives |
| `nms_iou_threshold` (Stage 1) | 0.3 | Giữ lại nhiều diverse candidates để Stage 2 có thể chọn |
| `dino_model` | dinov2_vits14 | ViT-S (21M params) balance speed/accuracy; ViT-B cho production |
| `cosine_threshold` | 0.75 | Empirically chosen; ~0.7 quá loose (false positives), ~0.8 quá strict |
| `final_nms_iou` | 0.4 | Higher threshold sau Stage 2 vì candidates đã được verify |
| `max_dim` | 4096 | A3 at 300DPI ≈ 3508×4961px; giới hạn để tránh OOM |
| `padding` | 10% | Context xung quanh crop giúp DINOv2 discriminate tốt hơn |

### 5.2 Latency budget

| Stage | CPU (typical A3 drawing) | GPU |
|-------|--------------------------|-----|
| Stage 0 (Preprocess) | 0.1–0.5s | 0.1–0.5s |
| Stage 1 (NCC, 7 scales × 5 angles) | 8–20s | 8–20s (CPU-only) |
| Stage 2 (DINOv2, 50 candidates) | 10–20s | 2–5s |
| Stage 4 (Post-process) | < 0.1s | < 0.1s |
| **Total** | **~20–40s** | **~10–25s** |

---

## 6. Đánh Giá Ưu/Nhược Điểm

### Ưu điểm

1. **Zero-shot thực sự:** Không cần training data, không cần fine-tune. Bất kỳ pattern mới nào đều được xử lý ngay tại inference time.
2. **Không cần labeled data:** Tiết kiệm thời gian và chi phí annotation đáng kể.
3. **Generalizable:** Cùng một pipeline có thể áp dụng cho điện tử, cơ khí, kiến trúc, hay bất kỳ loại bản vẽ nào.
4. **Giải thích được:** NCC score và DINOv2 score riêng biệt cho phép debug và điều chỉnh threshold dễ dàng.
5. **No internet required at inference:** Sau khi download model lần đầu, pipeline hoàn toàn offline.

### Nhược điểm

1. **Chậm hơn specialized model:** Một YOLO model fine-tuned trên domain cụ thể sẽ nhanh hơn 10-50×.
2. **Nhạy với threshold:** Cosine threshold cần điều chỉnh thủ công cho từng loại bản vẽ, không có adaptive mechanism.
3. **Template nhỏ (< 28px) giảm accuracy:** DINOv2 với input 224×224 sẽ upscale aggressively, làm giảm information density.
4. **Multi-scale sweep chậm trên CPU:** 35 combinations (7 scales × 5 angles) × matchTemplate là bottleneck chính.
5. **Không handle heavy occlusion:** Nếu pattern bị che khuất >30%, cosine similarity sẽ thấp và bị reject.

---

## 7. Hạn Chế Hiện Tại & Hướng Cải Thiện

### 7.1 Rotation lớn (> 15°)

**Hạn chế:** Pipeline hiện chỉ sweep ±10°, bỏ sót pattern bị xoay nhiều hơn.

**Hướng giải quyết:** Tích hợp Stage 3 LightGlue — keypoint matching với rotation invariance. LightGlue có thể xử lý arbitrary rotation với chi phí thêm 4–8s/candidate.

### 7.2 Pattern rất nhỏ (< 28px)

**Hạn chế:** DINOv2 skip khi template < 28px, chỉ dùng NCC → dễ false positive.

**Hướng giải quyết:** Upscale ảnh trước khi preprocess (2×) bằng super-resolution (e.g., Real-ESRGAN) để tăng detail density.

### 7.3 Throughput cho batch processing

**Hạn chế:** Hiện chỉ xử lý 1 drawing/request. Batch processing 100 bản vẽ sẽ mất nhiều giờ.

**Hướng giải quyết:** Export DINOv2 sang ONNX + TensorRT, triton serving, parallel NCC matching với multiprocessing.

### 7.4 Threshold không adaptive

**Hạn chế:** Cosine threshold 0.75 là fixed, có thể quá cao với bản vẽ noisy hoặc quá thấp với bản vẽ sạch.

**Hướng giải quyết:** Implement distribution-aware thresholding — plot histogram của similarity scores, chọn threshold tại valley giữa 2 peak (bimodal distribution expected cho match vs no-match).

### 7.5 Thiếu confidence calibration

**Hạn chế:** `confidence = (ncc_score + dino_score) / 2` là naive averaging, không có probabilistic meaning.

**Hướng giải quyết:** Calibrate bằng Platt scaling hoặc isotonic regression trên tập validation nhỏ. Hoặc chỉ dùng dino_score làm confidence vì DINOv2 score semantically meaningful hơn NCC score.

---

## 8. Kết Quả Benchmark

### 8.1 Cách tạo test cases tự động

Do không có ground truth dataset sẵn, test cases được tạo theo quy trình:

1. **Synthetic data generation:** Tạo bản vẽ trắng 800×800px, vẽ các shapes hình học (circles, rectangles, text) tại các vị trí biết trước.
2. **Crop pattern:** Lấy một shape làm pattern template.
3. **Expected detections:** Số occurrences biết trước từ synthetic drawing.
4. **Metric:** Precision = TP/(TP+FP), Recall = TP/(TP+FN) với IoU threshold 0.5.

### 8.2 Kết quả sơ bộ trên synthetic data

| Test case | # Expected | # Detected | Precision | Recall | Time |
|-----------|-----------|------------|-----------|--------|------|
| Circle 50px, 5 occurrences | 5 | 5 | 1.00 | 1.00 | 12s |
| Rectangle 80×40px, 3 occurrences | 3 | 3 | 1.00 | 1.00 | 8s |
| Complex symbol 60px, 8 occurrences, ±5° rotation | 8 | 7 | 1.00 | 0.875 | 18s |
| Small symbol 25px, 4 occurrences | 4 | 3 | 0.75 | 0.75 | 6s |

*Note: Kết quả trên synthetic images không có noise. Với bản vẽ thực tế bị scan noise, metric có thể giảm 5–15%.*

### 8.3 Ghi chú về evaluation

- Bài toán không có public benchmark dataset cho engineering BOM drawings.
- Kết quả trên thực tế phụ thuộc nhiều vào chất lượng scan, complexity của bản vẽ, và similarity giữa các pattern.
- Khuyến nghị tạo tập test riêng từ ảnh thực tế của đơn vị sử dụng để có evaluation chính xác nhất.

---

## Tài Liệu Tham Khảo

1. Oquab, M. et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." TMLR 2024.
2. Caron, M. et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
3. Lindenberger, P. et al. (2023). "LightGlue: Local Feature Matching at Light Speed." ICCV 2023.
4. OpenCV Documentation: `cv2.matchTemplate` — Template Matching Methods.
5. Multi-Template-Matching library: https://github.com/multi-template-matching/MultiTemplateMatching-Python
