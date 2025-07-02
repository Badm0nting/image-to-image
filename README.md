# 🧠 Pix2Pix Image-to-Image Translation with TensorFlow

This project implements a Pix2Pix conditional GAN (cGAN) using TensorFlow to perform image-to-image translation. Specifically, it uses the **CMP Facades dataset** to learn how to generate building facade photographs from architectural labels.

---

## 🎯 Objective

- Learn and implement a cGAN-based Pix2Pix model using TensorFlow 2.x
- Train the model on the CMP Facades dataset (label → photo)
- Generate and visualize realistic facade images from label maps

---

## 📂 Dataset

**Source**: [Berkeley Pix2Pix Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

- 📦 `train/` — 400 image pairs
- 🧪 `test/` — 106 image pairs
- 📊 `val/` (optional) — 100 image pairs
- Each `.jpg` is a 256×512 image:  
  Left = label map, Right = real facade

### ✅ Dataset Download (included in code):

```python
tf.keras.utils.get_file(
    "facades.tar.gz",
    origin="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
)
```
## 🏗️ Model Architecture
Generator: U-Net
Downsampling + upsampling

Skip connections

## Discriminator: PatchGAN
Classifies 70x70 image patches as real or fake

## 🧪 Training Configuration
| Setting    | Value                          |
| ---------- | ------------------------------ |
| Epochs     | 30                             |
| Batch Size | 1                              |
| Image Size | 256×256                        |
| Optimizer  | Adam (lr=2e-4, β1=0.5)         |
| Loss       | Binary cross-entropy + L1 loss |
| Framework  | TensorFlow 2.x                 |

## 📈 Output
For each test image:

🖼️ Input label

📷 Ground truth facade

🧠 Predicted image from generator

## 📚 References
- [TensorFlow Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)
- [Original Pix2Pix Paper (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)

## ✍️ Author
Indraneal Sajjankar
