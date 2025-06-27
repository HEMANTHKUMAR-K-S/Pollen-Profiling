
# 🌼 Pollen's Profiling – Automated Classification of Pollen Grains

Pollen's Profiling is an intelligent web application that automates the classification of pollen grains using deep learning. Designed for researchers, environmentalists, healthcare professionals, and agricultural scientists, it leverages a trained Convolutional Neural Network (CNN) to accurately identify pollen types from microscopic images.

---

## 🧠 Project Overview

🔬 **Problem**: Manual pollen classification is time-consuming, error-prone, and not scalable.  
🤖 **Solution**: Use deep learning to classify images of pollen grains with high accuracy via a modern web interface.

---

## 📁 Project Structure

```

POLLEN\_GRAIN/
├── data/                  # Training images organized by class
├── flask/                 # Flask backend and templates
│   ├── static/
│   ├── templates/
│   ├── app.py             # Flask app
│   ├── requirements.txt   # Flask dependencies
│   └── Procfile           # For deployment
├── uploads/               # Stores uploaded test images           
├── model.h5               # Final model used in production
├── train\_model.py         # Model training script
├── class\_map.json         # Maps class indices to pollen names
└── README.md              # This file

```

---

## 🚀 How to Run the Project Locally

### 🔧 1. Train the Model (if not using pre-trained model)

Make sure your dataset is downloaded and structured like:

Then run:

```bash
python train_model.py
````

This will generate:

* `model.h5`
* `class_map.json`

### ⚙️ 2. Start the Flask App

```bash
cd flask
pip install -r requirements.txt
python app.py
```

## 💡 Features

* 📷 Upload pollen grain images
* 🤖 Predict pollen type using trained CNN
* 🌐 Lightweight Flask-based backend
* 🎨 Clean, animated frontend (HTML + CSS)
* 💾 Easy-to-train model (Keras, TensorFlow)
* 📦 Ready for deployment on platforms like Render

---

## 🌍 Use Case Scenarios

* **🌱 Agriculture**: Improve crop management by classifying pollen for breeding.
* **🏥 Healthcare**: Assist allergists in identifying pollen types.
* **🌳 Environment**: Monitor biodiversity and ecosystem changes.

---

## 📸 Demo Screenshots

### 🖥️ Home Page

![image](https://github.com/user-attachments/assets/88f19976-af58-4e2c-ad19-97096c638e96)

### ✅ Prediction Result Page

![image](https://github.com/user-attachments/assets/57680045-bbf1-4768-8eaa-940742dce5db)

---

## 🔍 Tech Stack

| Layer      | Technology         |
| ---------- | ------------------ |
| Frontend   | HTML, CSS, JS      |
| Backend    | Python, Flask      |
| ML Model   | TensorFlow, Keras  |
| Deployment | Render / Localhost |

---

## 🧪 Testing

* Upload a pollen image via the web interface.
* Check the predicted class with confidence score.
* Try uploading different pollen types to evaluate model generalization.

---

## 🙌 Acknowledgments

* TensorFlow & Keras Documentation
* Flask Community
* Kaggle Datasets
* Open Source Contributors

---

## ✨ Author

**K S Hemanthkumar**
🔗 [GitHub](https://github.com/HEMANTHKUMAR-K-S)

