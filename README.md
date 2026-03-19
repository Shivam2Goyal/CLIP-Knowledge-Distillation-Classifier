# Age Classification: Mitigating Bias via ResNet-18 + CLIP Distillation

This project addresses a binary classification task to categorize face images into **Young (0)** and **Old (1)**. While seemingly straightforward, the core challenge lies in overcoming significant dataset bias.

---

## ⚠️ The Challenge: Spurious Correlations

The provided dataset contains a strong **spurious correlation** between gender and age:
* **Young faces** are predominantly female.
* **Old faces** are predominantly male.

A standard supervised model naturally gravitates toward **gender cues** (a shortcut) rather than learning the physiological features of aging. This results in high training accuracy but poor generalization on balanced or out-of-distribution data.

---

## 💡 The Solution: Representation Learning

To force the model to learn robust visual features rather than shortcuts, I implemented a multi-stage pipeline centered around **Knowledge Distillation**.

### 1. CLIP Feature Distillation
Instead of starting with raw classification, I used a **CLIP ViT-L/14** model as a teacher.
* **Process:** The ResNet-18 backbone was trained to minimize the **Cosine Similarity Loss** between its output embeddings and CLIP's rich, general-purpose embeddings.
* **Goal:** Leverage CLIP's pre-trained understanding of "human features" to initialize the backbone with unbiased representations.



### 2. Robust Classification Training
Once the backbone was "primed," I trained the classifier using several regularization techniques:
* **CutMix Augmentation:** Forces the model to recognize objects from partial views, preventing reliance on specific local pixels.
* **Exponential Moving Average (EMA):** Maintains a "shadow" copy of weights to smooth out training noise and improve stability.
* **OneCycleLR & Label Smoothing:** Optimizes convergence speed and prevents the model from becoming overconfident in biased labels.

### 3. Final Generalization Phase
* **SWA (Stochastic Weight Averaging):** Averaging multiple points along the trajectory of SGD to find a wider, more general local minimum.
* **Full Data Refinement:** Final training pass on the combined training and validation sets.

---

## 🧠 Model Architecture

* **Backbone:** ResNet-18 (Custom initialized via distillation)
* **Classification Head:**
    * BatchNorm $\rightarrow$ Dropout $\rightarrow$ Linear (512 to 256) $\rightarrow$ ReLU $\rightarrow$ Linear (256 to 2)

**Pipeline Flow:**
`Image` → `ResNet-18` → `512-d Feature Vector` → `MLP Head` → `Logits (Young/Old)`

---

## 📊 Performance Comparison

| Model Phase | Training Accuracy | Validation Accuracy |
| :--- | :---: | :---: |
| **Phase I (Baseline)** | ~92.00% | ~82.00% |
| **Phase II (Final Distilled)** | **99.17%** | **91.79%** |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install torch torchvision tqdm pillow
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```
2. Execution
To train the model from scratch:

```bash
python train.py
```
To evaluate the final weights:

```bash
python evaluate_submission_student.py --model_path b23cm1036.pth --model_file b23cm1036.py --data_dir dataset
```
---
## 📁 Project Structure
```
Plaintext
├── b23cm1036.py           # Model architecture & class definitions
├── train.py               # Full training & distillation pipeline
├── dataset/               # Organized into /train and /valid
└── README.md              # Documentation
```
### 🤝 Key Takeaways
* Accuracy is a Lie: High metrics often mask "shortcut learning" if the dataset is biased.

* Foundation Models Help: Using CLIP as a teacher provides a "semantic compass" that raw labels cannot provide.

* Generalization > Optimization: Techniques like SWA and EMA are essential for moving beyond the specific noise of a single training set.
