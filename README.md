# 🐟 Multiclass Fish Image Classification

This project uses **Deep Learning (CNN and Transfer Learning)** to classify fish images into multiple categories and deploys a **Streamlit web app** for real-time predictions.

## 🚀 Project Overview
Fish species classification is crucial for biodiversity research, fisheries management, and automated seafood processing.  
This project builds and evaluates CNN-based models to identify fish species from images and provides an interactive interface for predictions.

## 🧠 Skills Demonstrated
- Deep Learning with TensorFlow/Keras  
- Data Preprocessing & Augmentation  
- Transfer Learning (ResNet50, MobileNet, etc.)  
- Model Evaluation & Visualization  
- Streamlit Deployment  

## 📊 Dataset
The dataset contains fish images categorized by species.  
It is preprocessed and augmented using **ImageDataGenerator**.  
Images are resized to `(224, 224)` and normalized to `[0, 1]`.

## 🏗 Approach
1. **Data Preprocessing**
   - Rescale and augment images.
2. **Model Training**
   - Build a CNN from scratch.
   - Fine-tune multiple pre-trained models (ResNet50, VGG16, MobileNetV2, InceptionV3, EfficientNetB0).
3. **Evaluation**
   - Compare accuracy, precision, recall, F1-score.
   - Visualize accuracy/loss curves.
4. **Deployment**
   - Build a Streamlit app to upload an image and predict its species.

## 💻 Streamlit App
To run locally:
```bash
pip install -r requirements.txt
streamlit run TestTen_Fish.py
```
Upload any fish image and get:
- Predicted species name  
- Model confidence score  
- Confidence breakdown for all classes  

## 📈 Model Used
The best-performing model was **ResNet50** fine-tuned on the dataset and saved as `resnet50_trained_model.h5`.

## 📂 Repository Structure
```
├── CNN_FishImagef.ipynb       # Model training
├── TestTen_Fish.py            # Streamlit deployment
├── resnet50_trained_model.h5  # Trained model (add manually)
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── Project_Report.pdf         # Detailed project report
```

## 🔍 Evaluation Metrics
| Metric | CNN | ResNet50 | MobileNetV2 | EfficientNetB0 |
|:-------:|:---:|:---------:|:-------------:|:---------------:|
| Accuracy | 87% | **96%** | 94% | 95% |
| Precision | 0.85 | 0.95 | 0.93 | 0.94 |
| Recall | 0.86 | 0.96 | 0.93 | 0.95 |

## 🌐 Deployment
You can deploy this Streamlit app on:
- Streamlit Cloud
- Render
- Hugging Face Spaces

## 📜 License
This project is released under the MIT License.
