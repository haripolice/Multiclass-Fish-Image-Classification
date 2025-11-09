# 🐟 Multiclass Fish Image Classification - Project Report

## 🎯 Project Overview
This project aims to classify fish images into multiple categories using **Deep Learning** models.  
It includes model training, evaluation, and deployment using **Streamlit** for real-time predictions.

---

## 🧠 Skills Demonstrated
- Deep Learning  
- Python  
- TensorFlow / Keras  
- Streamlit  
- Data Preprocessing & Augmentation  
- Transfer Learning  
- Model Evaluation  
- Model Deployment  

---

## 🌍 Domain
**Image Classification**  

The model identifies the species of fish from uploaded images, helping automate classification tasks for fisheries and marine biology research.

---

## 📋 Problem Statement
Develop a multiclass image classifier to categorize fish species using CNNs and transfer learning.  
The goal is to create a model that achieves high accuracy and can be deployed as a web application.

---

## 💼 Business Use Cases
- **Enhanced Accuracy:** Identify the best-performing architecture for fish classification.  
- **Deployment Ready:** Provide an interactive web interface for predictions.  
- **Model Comparison:** Evaluate and compare multiple models to determine optimal performance.

---

## ⚙️ Approach

### 1. Data Preprocessing and Augmentation
- Rescale images to the `[0, 1]` range.  
- Apply rotation, zoom, and flipping for augmentation.  
- Use `ImageDataGenerator` for efficient batch loading.

### 2. Model Training
- Train a custom CNN model from scratch.  
- Experiment with pre-trained models like:
  - VGG16  
  - ResNet50  
  - MobileNetV2  
  - InceptionV3  
  - EfficientNetB0  
- Fine-tune models for better performance.  
- Save the best model (`.h5` format).

### 3. Model Evaluation
- Evaluate models using accuracy, precision, recall, and F1-score.  
- Plot training vs. validation accuracy and loss.  
- Generate confusion matrices for visual insight.

### 4. Deployment
- Use **Streamlit** to build an interactive app.  
- Users can upload fish images to get real-time predictions and confidence scores.

---

## 🧪 Evaluation Metrics

| Metric | CNN | ResNet50 | MobileNetV2 | EfficientNetB0 |
|:-------:|:---:|:---------:|:-------------:|:---------------:|
| Accuracy | 87% | **96%** | 94% | 95% |
| Precision | 0.85 | 0.95 | 0.93 | 0.94 |
| Recall | 0.86 | 0.96 | 0.93 | 0.95 |

---

## 🧩 Deliverables
- Trained models (.h5)  
- Streamlit app (`TestTen_Fish.py`)  
- Training notebook (`CNN_FishImagef.ipynb`)  
- Comparison report and plots  
- GitHub repository with complete documentation  

---

## 📦 Deployment
Deployed as a Streamlit app where users can:
1. Upload a fish image (`.jpg`, `.jpeg`, `.png`).  
2. View predicted class and confidence score.  
3. See per-class confidence breakdown.

---

## 🧰 Tools & Libraries
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Pillow  
- Streamlit  

---

## 📜 Coding Standards
- Follows [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines.  
- Modular, maintainable, and portable code.

---

## 🧾 Author
**Playboy**  
_Data Scientist_  

---

## 📚 References
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)  
- TensorFlow Tutorials: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)  

---

## 🪄 License
This project is released under the **MIT License**.
