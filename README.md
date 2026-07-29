# Potato Leaf Disease Classifier

A binary image classifier that distinguishes between **Early Blight** and **Healthy** potato leaves using transfer learning with MobileNetV3Small.

---

## Classes

| Label | Description |
|---|---|
| `Early_Blight` | Leaf infected by *Alternaria solani* |
| `Healthy` | No visible disease |

---

## Dataset

Two Kaggle datasets were combined to address the imbalance in healthy potato leaf samples:

- [`faysalmiah1721758/potato-dataset`](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) — primary source, includes both Early Blight and Healthy splits
- [`nirmalsankalana/potato-leaf-healthy-and-late-blight`](https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-healthy-and-late-blight) — supplementary source used to augment the Healthy class

After merging, each class was capped and split as follows:

| Split | Samples per class |
|---|---|
| Train | 450 |
| Test | 25 |
| Validation | 25 |

Images were resized to **224 × 224 px**.

---

## Model Architecture

**Base:** MobileNetV3Small pretrained on ImageNet (frozen, feature extraction mode)

**Head:**
- `GlobalAveragePooling2D`
- `Dense(256, activation='relu')`
- `Dropout(0.2)`
- `Dense(2, activation='softmax')`

The base model weights were kept frozen throughout training (no fine-tuning phase).

**Data augmentation applied during training:**
- Random rotation (±30%)
- Random zoom (±20%)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 30 |
| Max epochs | 50 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Seed | 30 |

**Callbacks:**
- `ModelCheckpoint` — saves best model by `val_accuracy`
- `EarlyStopping` — patience 8, monitors `val_loss`
- `ReduceLROnPlateau` — factor 0.3, patience 4, min LR 1e-7

---

## Saved Model

The best checkpoint is saved as:

```
tl_feature_extraction_best.keras
```

---

## Streamlit App

A simple web interface is provided in `app.py`.

### Requirements

```bash
pip install streamlit tensorflow pillow
```

### Run

Place `tl_feature_extraction_best.keras` in the same directory as `app.py`, then:

```bash
streamlit run app.py
```

Upload a potato leaf image and the app will return the predicted class along with class probabilities.

---

## Dependencies

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
streamlit
pillow
```
