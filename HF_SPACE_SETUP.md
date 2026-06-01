# HuggingFace Spaces Deployment Guide

## 🚀 Quick Setup

### 1. Create Space on HuggingFace

1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   ```
   Space name: Pattern_detection_in_technical_drawings
   License: MIT
   SDK: Docker ⭐ (IMPORTANT)
   Visibility: Public
   ```
4. Click **"Create space"**

---

## 2. Get Clone Command

After creation, HuggingFace shows:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/Pattern_detection_in_technical_drawings
cd Pattern_detection_in_technical_drawings
```

---

## 3. Push Code

```bash
# Clone GitHub repo
git clone https://github.com/DuyhocAI/Pattern_detection_in_technical_drawings.git
cd Pattern_detection_in_technical_drawings

# Add HF remote
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/Pattern_detection_in_technical_drawings

# Push to HF
git push huggingface main
```

---

## 📊 Hardware Requirements

### Minimum (Free T4 GPU)
- **VRAM:** 4GB (DINOv2 ViT-S/14)
- **CPU:** 2 cores
- **Disk:** 20GB (model cache)
- **Runtime:** ~25-30 seconds per drawing

### Recommended (Pro/Enterprise)
- **GPU:** RTX 3060 12GB or better
- **Disk:** 30GB (DINOv2 + Qwen2-VL optional)
- **Runtime:** ~15-20 seconds per drawing

### CPU-only (slow, not recommended)
- **CPU:** 4+ cores
- **RAM:** 8GB
- **Runtime:** 60-120 seconds per drawing

---

## ⚙️ Configuration

### Space Settings (HF Web UI)

**Settings → Hardware:**
- Select: `T4 GPU` (free) or higher
- Persistent Storage: Optional (for caching models)

**Settings → Repository:**
- Visibility: Public
- Private: No

---

## 🐳 Docker Details

### Base Image
```dockerfile
FROM python:3.11-slim
```
- Minimal, ~100MB
- No CUDA (CPU-only PyTorch for HF free tier)

### Pre-downloaded at Build
- **DINOv2 ViT-S/14** (~86 MB)
- Loaded on first request

### Lazy-loaded on First Use
- **Qwen2-VL-2B** (~5 GB) — only if `use_vlm=True`

### Build Time
- ~3-5 minutes on HF
- Includes dependency installation + DINOv2 download

---

## 📋 Supported File Types

**Pattern Template:** PNG, JPG, JPEG, BMP
**Engineering Drawing:** PNG, JPG, JPEG, BMP

**Recommended Size:**
- Pattern: 30×50 to 200×300 pixels
- Drawing: A3 size (1200×1700+) or larger

---

## 🔧 Troubleshooting

### Space takes too long to build
- **Normal:** First build is slow (3-5 min)
- **Fix:** Rebuild will be faster (Docker caching)

### Out of memory
- **Symptom:** "CUDA out of memory" or process killed
- **Fix:** Disable VLM or upgrade to paid tier

### Models not downloading
- **Symptom:** "torch.hub.load timeout"
- **Fix:** Wait for build to complete, models download at build time

### Port 7860 not responding
- **Check:** Logs in HF Space settings
- **Common:** App crashed, check error output

---

## 📈 Performance on HF Free Tier

| Metric | Value |
|--------|-------|
| Build time | 3-5 min |
| First request | 30-40s (cold start) |
| Subsequent requests | 20-30s |
| Idle timeout | 48 hours (HF policy) |
| Max upload size | 25 MB |

---

## 🎯 Features Available on HF

✅ Pattern upload
✅ Drawing upload
✅ Real-time detection
✅ Visualization output
✅ CSV export
✅ Web UI (FastAPI)
✅ Auto-tune mode
⚠️ VLM filter (slow on free tier, ~1 min per crop)

---

## 📚 Links

- **GitHub:** https://github.com/DuyhocAI/Pattern_detection_in_technical_drawings
- **System Spec:** See `design_spec/system_spec.html`
- **Model Survey:** See `design_spec/model_survey.md`
- **Docker Guide:** See `DOCKER.md`

---

## ⚡ Quick Commands

```bash
# Test locally before pushing to HF
docker build -t bom-detector:latest .
docker run -p 8000:7860 bom-detector:latest

# Then push to HF
git push huggingface main
```

---

## 🚀 After Deployment

1. HF will auto-build (3-5 min)
2. Space will be live at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/Pattern_detection_in_technical_drawings
   ```
3. Anyone can use without login

---

## 💡 Tips

- **First-time cold start:** ~30-40 seconds (normal)
- **Subsequent runs:** ~20-25 seconds (faster)
- **Idle cleanup:** HF pauses spaces after 48 hours of inactivity
- **Cost:** FREE on T4 GPU tier (limited hours/month)

---

**Ready to deploy? Create the Space and push the code!** 🎉
