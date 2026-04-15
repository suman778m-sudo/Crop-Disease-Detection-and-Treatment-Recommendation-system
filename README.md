 Crop Disease Detection and Treatment Recommendation System

An intelligent deep learning-based system that automatically identifies plant diseases from leaf images and provides actionable treatment recommendations — helping farmers and agronomists protect crops and maximize yield.

---

 Overview

This project uses a Convolutional Neural Network (CNN) trained on the **PlantVillage dataset** to classify plant leaf images into healthy or diseased categories. For each detected disease, the system provides detailed information on **probable causes** and **recommended solutions**.

-  **Final Validation Accuracy: 94.85%**
-  **Final Validation Loss: 0.1712**
-  Trained on **Google Colab** with a T4 GPU

---

 Supported Plants & Diseases

The model can detect diseases across **14 plant species**, covering **38 classes** in total (including healthy variants):

| Plant | Diseases Detected |
|---|---|
|  Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
|  Blueberry | Healthy |
|  Cherry | Powdery Mildew, Healthy |
|  Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
|  Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
|  Orange | Haunglongbing (Citrus Greening) |
|  Peach | Bacterial Spot, Healthy |
|  Bell Pepper | Bacterial Spot, Healthy |
|  Potato | Early Blight, Late Blight, Healthy |
|  Raspberry | Healthy |
|  Soybean | Healthy |
|  Squash | Powdery Mildew |
|  Strawberry | Leaf Scorch, Healthy |
|  Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

 Model Architecture

- **Base Model:** Custom CNN (trained from scratch)
- **Framework:** TensorFlow / Keras
- **Input Size:** 128×128 RGB images
- **Output:** Softmax classification over 38 disease/healthy classes
- **Saved Format:** `.keras`

---

 Repository Structure

```
├── CV_Project.ipynb            # Main training notebook (Google Colab)
├── class_names_enhanced.json   # Disease metadata with causes & solutions
├── plant_disease_model.keras   # Trained model weights (generated after training)
└── README.md
```

---

 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/suman778m-sudo/Crop-Disease-Detection-and-Treatment-Recommendation-system.git
cd Crop-Disease-Detection-and-Treatment-Recommendation-system
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib pillow kaggle
```

### 3. Download the Dataset

This project uses the **PlantVillage Dataset** from Kaggle. Set up your Kaggle API credentials (`kaggle.json`) and the notebook will handle the download automatically.

### 4. Train the Model

Open `CV_Project.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells. A GPU runtime (T4 or better) is recommended.

The notebook will:
- Download and preprocess the dataset
- Train the CNN model
- Evaluate on the validation set
- Save `plant_disease_model.keras` and `class_names.json`

### 5. Run Predictions

Load the saved model and make predictions on new leaf images:

```python
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class metadata
model = tf.keras.models.load_model('plant_disease_model.keras')
with open('class_names_enhanced.json') as f:
    class_data = json.load(f)

# Preprocess image
img = Image.open('leaf.jpg').resize((128, 128))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# Predict
pred = model.predict(img_array)
class_names = list(class_data.keys())
predicted_class = class_names[np.argmax(pred)]

info = class_data[predicted_class]
print(f"Plant   : {info['plant']}")
print(f"Disease : {info['disease']}")
print(f"\nCauses  : {info['reasons'][0]}")
print(f"Solution: {info['solutions'][0]}")
```

---

 Training Results

| Metric | Value |
|---|---|
| Validation Accuracy | **94.85%** |
| Validation Loss | **0.1712** |
| Training Platform | Google Colab (T4 GPU) |
| Epochs | 10+ |

---

 How the Recommendation Works

Each class in `class_names_enhanced.json` contains:
- **`plant`** – Crop name
- **`disease`** – Disease name (or "None – Healthy")
- **`reasons`** – List of probable causes for the disease
- **`solutions`** – Actionable treatment and prevention steps

This structured metadata allows the system to provide instant, expert-level guidance alongside each prediction.

---

 Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- Google Colab
- Kaggle API (PlantVillage Dataset)

---

 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the model, add more diseases, or build a web/mobile frontend.

---

 License

This project is open-source and available under the [MIT License](LICENSE).

---

 Author

**suman778m-sudo**  
[GitHub Profile](https://github.com/suman778m-sudo)
