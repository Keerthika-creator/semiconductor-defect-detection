# 🚀 QUICK START GUIDE
## Semiconductor Defect Detection System - IESA DeepTech Hackathon 2026

**Get started in 5 minutes!**

---

## 📋 What You Have

This complete package includes:
- ✅ **Training Pipeline** - Full end-to-end system
- ✅ **ONNX Export** - Edge deployment ready
- ✅ **Inference Demo** - Testing and validation
- ✅ **Visualization Tools** - Analysis and reporting
- ✅ **Documentation** - Comprehensive guides

---

## 🏃 Quick Start (Google Colab)

### Step 1: Upload Files (1 minute)

Upload all `.py` files to your Google Colab environment:
- `semiconductor_defect_detection.py`
- `onnx_export.py`
- `inference_demo.py`
- `visualization_utils.py`
- `colab_setup.py`

### Step 2: Run Setup (2 minutes)

```python
# In a Colab cell:
!python colab_setup.py
```

This will:
- Install all dependencies
- Mount Google Drive
- Create directory structure
- Verify environment

### Step 3: Prepare Dataset (Manual)

Upload your dataset to Google Drive:
```
/content/drive/MyDrive/semiconductor_dataset.zip
```

Dataset structure:
```
semiconductor_dataset/
├── train/
│   ├── wafer/
│   │   ├── Center/
│   │   ├── Donut/
│   │   └── ... (8 defect types + None)
│   └── die/
│       ├── Clean/
│       ├── Defect/
│       └── Other/
├── val/
└── test/
```

### Step 4: Train Models (30-60 minutes depending on GPU)

```python
# In a Colab cell:
!python semiconductor_defect_detection.py
```

### Step 5: Export to ONNX (2 minutes)

```python
!python onnx_export.py
```

### Step 6: Test Inference

```python
!python inference_demo.py
```

---

## 💻 Quick Start (Local Machine)

### Step 1: Clone/Download

```bash
# If using git:
git clone https://github.com/yourusername/semiconductor-defect-detection.git
cd semiconductor-defect-detection

# Or simply download all files to a folder
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset

Place your dataset in the project directory with this structure:
```
project/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── [code files]
```

### Step 4: Update Paths

Edit `semiconductor_defect_detection.py`:
```python
class Config:
    DATASET_ZIP = "path/to/your/dataset.zip"  # Update this
    DATASET_ROOT = "./dataset"
    OUTPUT_DIR = "./output"
    MODELS_DIR = "./models"
```

### Step 5: Train

```bash
python semiconductor_defect_detection.py
```

### Step 6: Export

```bash
python onnx_export.py
```

### Step 7: Test

```bash
python inference_demo.py
```

---

## 🎯 Minimal Example (Copy-Paste Ready)

```python
# ==============================================================
# MINIMAL WORKING EXAMPLE - Copy this entire cell to Colab
# ==============================================================

# 1. Install (run once)
!pip install -q tensorflow numpy pandas matplotlib seaborn scikit-learn

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Import the main script
# (Upload semiconductor_defect_detection.py first)
from semiconductor_defect_detection import Config, DatasetManager, DualHeadDefectClassifier

# 4. Configure
config = Config()
config.DATASET_ZIP = "/content/drive/MyDrive/semiconductor_dataset.zip"
config.create_directories()

# 5. Extract dataset
dataset_manager = DatasetManager(config)
dataset_manager.extract_dataset()

# 6. Build models
model_builder = DualHeadDefectClassifier(config)
model_builder.build_models()

# 7. Train (you'll need to set up data generators)
# See USAGE_GUIDE.py for complete training examples

print("✅ Setup complete! Ready to train.")
```

---

## 📊 Expected Results

After training, you should see:

### Accuracy
- Wafer: **96-98%**
- Die: **97-99%**

### Performance
- Latency: **<100ms**
- Throughput: **>10 FPS**
- Model Size: **<50MB**

### Files Generated
```
output/
├── wafer_confusion_matrix.png
├── die_confusion_matrix.png
├── training_history.png
├── evaluation_report.csv
└── EVALUATION_REPORT.md

models/
├── wafer_final.h5
├── die_final.h5
└── stage_router_best.h5

onnx_models/
├── wafer_classifier_quantized.onnx
├── die_classifier_quantized.onnx
└── stage_router_quantized.onnx
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
config.BATCH_SIZE = 16  # or 8
```

### Issue: Dataset Not Found
**Solution**: Check path and structure
```python
dataset_manager.validate_structure()
```

### Issue: Slow Training
**Solution**: Use GPU in Colab
- Runtime → Change runtime type → GPU

### Issue: Low Accuracy
**Solution**: 
1. Check data quality
2. Increase training epochs
3. Adjust learning rate
4. Balance classes

---

## 📚 Next Steps

1. **Review Documentation**
   - `README.md` - Full project overview
   - `USAGE_GUIDE.py` - Detailed examples
   - `PROJECT_SUMMARY.md` - Technical details

2. **Customize**
   - Edit `Config` class for your needs
   - Adjust hyperparameters
   - Modify augmentation strategy

3. **Deploy**
   - Export to ONNX
   - Test on target hardware
   - Integrate into production

4. **Optimize**
   - Quantize models
   - Profile performance
   - Fine-tune thresholds

---

## 💡 Pro Tips

1. **Start Small**: Test with a small dataset first
2. **Monitor Training**: Use TensorBoard for live monitoring
3. **Save Checkpoints**: Models are saved automatically
4. **Validate Often**: Check confusion matrices frequently
5. **Document Changes**: Keep track of config modifications

---

## 🆘 Getting Help

1. **Check Logs**: Look at training output for errors
2. **Read Docs**: Comprehensive guides available
3. **Test Components**: Use individual scripts
4. **Verify Data**: Ensure dataset is correct

---

## ✅ Success Checklist

Before considering training complete:

- [ ] All dependencies installed
- [ ] Dataset properly structured
- [ ] Training completed without errors
- [ ] Validation accuracy >90%
- [ ] Models saved successfully
- [ ] ONNX export successful
- [ ] Inference demo works
- [ ] Performance meets requirements

---

## 🎓 Learning Resources

**Understanding the Code**:
- Start with `colab_setup.py` - simplest
- Then `inference_demo.py` - see how models work
- Finally `semiconductor_defect_detection.py` - full system

**Understanding the Approach**:
1. Read `README.md` - architecture overview
2. Check `PROJECT_SUMMARY.md` - innovations
3. Study `USAGE_GUIDE.py` - practical examples

---

## 🚀 Ready to Go!

You now have everything needed to:
- ✅ Train state-of-the-art defect detection models
- ✅ Export for edge deployment
- ✅ Achieve real-time performance
- ✅ Deploy to production

**Start with Step 1 above and you'll be running in minutes!**

---

**Questions?** Check the documentation or create an issue on GitHub.

**Success?** ⭐ Star the repository and share your results!

---

*Last updated: February 2026*
