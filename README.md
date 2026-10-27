# Vehicle Type Classification using Computer Vision 🚗🚌🚚🏍️

A beginner-friendly Machine Learning project that classifies images of vehicles into four categories (**Car, Bus, Truck, Motorcycle**) using **Transfer Learning** with a pretrained **MobileNetV2** model and serves predictions via an interactive **Streamlit** web application.

---

## 📌 Project Overview

In Computer Vision, training an image classification model from scratch requires tens of thousands of labeled images and massive GPU compute power. **Transfer Learning** solves this by starting with a model (like **MobileNetV2**) that has already learned rich visual representations (edges, textures, shapes, silhouettes) from over 1.4 million images in the famous **ImageNet** benchmark.

By freezing the pretrained base and adding a custom lightweight classification head on top, we can train an accurate vehicle classifier locally in just a few minutes using standard CPU or GPU hardware!

### Target Classes (4 Categories)
1. **Car**
2. **Bus**
3. **Truck**
4. **Motorcycle**

---

## 📂 Project Structure

Here is how your project workspace is organized:

```text
MLPROJECT/
│
├── data/                               # Dataset directory organized by split & class
│   ├── train/                          # Images used to train the model (~70%)
│   │   ├── Bus/
│   │   ├── Car/
│   │   ├── Motorcycle/
│   │   └── Truck/
│   ├── val/                            # Images used during training to monitor overfitting (~15%)
│   │   ├── Bus/
│   │   ├── Car/
│   │   ├── Motorcycle/
│   │   └── Truck/
│   └── test/                           # Separate images used strictly for final evaluation (~15%)
│       ├── Bus/
│       ├── Car/
│       ├── Motorcycle/
│       └── Truck/
│
├── models/                             # Saved artifacts from training
│   ├── vehicle_classifier_mobilenetv2.keras   # Saved model weights & architecture
│   ├── class_indices.json             # Mapping between numerical IDs and class names
│   ├── evaluation_metrics.txt         # Text report containing Accuracy, Precision, Recall, F1
│   └── confusion_matrix.png           # Visual heatmap of actual vs. predicted labels
│
├── test_samples/                       # Sample vehicle images for quick testing in the web app
│
├── app.py                              # Streamlit web application for image upload & prediction
├── train.py                            # Model building, transfer learning, training & evaluation
├── download_data.py                    # Helper script for dataset status check & sample data setup
├── requirements.txt                    # List of required Python dependencies
└── README.md                           # Step-by-step project documentation
```

### Explanation of Core Files

- **`requirements.txt`**: Specifies all required Python packages (`tensorflow`, `keras`, `streamlit`, `scikit-learn`, `pillow`, `numpy`, `matplotlib`).
- **`download_data.py`**: Validates whether your image folders are properly populated, explains how to obtain public vehicle datasets, and includes a one-command `--create-sample` feature to test your pipeline right away.
- **`train.py`**:
  1. Loads images from `data/train` and `data/val` at 224x224 resolution.
  2. Applies data augmentation (random horizontal flips, subtle rotations, and zooms) to prevent overfitting.
  3. Attaches the pretrained **MobileNetV2** base and freezes its weights.
  4. Trains a custom Dense classification head on your vehicle images.
  5. Evaluates model performance on the validation set using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and a **Confusion Matrix**.
  6. Saves the best model to `models/vehicle_classifier_mobilenetv2.keras`.
- **`app.py`**: A clean web application built with Streamlit where you can drag and drop any vehicle picture to see the predicted class along with confidence percentage and a probability breakdown.

---

## 🚀 Step-by-Step Getting Started Guide (Windows)

### Step 1: Open Terminal in Project Folder
Open **PowerShell** or **Command Prompt** and navigate to your project directory:
```powershell
cd c:\Users\kalya\OneDrive\Desktop\MLPROJECT
```

### Step 2: Install Dependencies
Install the required packages listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*(If you already have TensorFlow installed, this will quickly verify existing packages and install Streamlit).*

---

### Step 3: Dataset Setup

You have two options to set up data:

#### Option A: Quick Starter Dataset (For Pipeline Testing)
To verify that everything works end-to-end on your computer, run:
```powershell
python download_data.py --create-sample
```
This automatically generates clean starter images across all 4 categories in `data/train/`, `data/val/`, and `data/test/`, plus test sample images in `test_samples/`.

#### Option B: Using a Public Vehicle Dataset (For Real-World Training)
For high-accuracy real-world vehicle classification:
1. Download a public 4-class vehicle dataset:
   - **Kaggle Vehicle Type Recognition Dataset**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/iamsouravbanerjee/vehicle-type-recognition)
   - Or **Kaggle (brj090)**: [Vehicle Type Recognition](https://www.kaggle.com/datasets/brj090/vehicle-type-recognition)
2. Extract the downloaded dataset. If the images are in an unorganized folder, let the helper script split them automatically:
   ```powershell
   python download_data.py --organize "C:\path\to\extracted_folder"
   ```
   Or place them manually into:
   - `data/train/<ClassName>` (~70% of images)
   - `data/val/<ClassName>` (~15% of images)
   - `data/test/<ClassName>` (~15% of images)
3. Check your dataset folder status at any time:
   ```powershell
   python download_data.py --status
   ```

---

### Step 4: Train and Evaluate the Model

Run the training script:
```powershell
python train.py --epochs 5 --batch-size 16
```

#### What happens during training:
1. Pretrained **MobileNetV2** weights from ImageNet are loaded and frozen.
2. The custom vehicle classification head is trained on your vehicle images.
3. Checkpoint callbacks monitor validation performance and save the best model weights to `models/vehicle_classifier_mobilenetv2.keras`.
4. The model evaluates the separate `data/test/` dataset (or `data/val/` if test is empty) and outputs:
   - **Overall Accuracy**: Fraction of test images correctly predicted.
   - **Macro Precision**: Average precision across all 4 classes.
   - **Macro Recall**: Average recall across all 4 classes.
   - **Macro F1-Score**: Harmonic mean of macro precision and recall.
   - **Weighted F1-Score**: F1-score weighted by the number of samples in each class.
   - **Confusion Matrix**: 4x4 matrix showing exact true vs. predicted counts.
5. All evaluation metrics are saved to `models/evaluation_metrics.txt` and plotted to `models/confusion_matrix.png`.

---

### Step 5: Run the Streamlit Web Application

Launch the interactive local web app:
```powershell
streamlit run app.py
```

Streamlit will print a local URL in your terminal:
```text
Local URL: http://localhost:8501
```
Open that address in your web browser!

#### Using the Web App:
1. Upload any vehicle image (`.jpg`, `.jpeg`, `.png`) or pick a sample from the left sidebar.
2. The app preprocesses the image to 224x224 and feeds it to your trained MobileNetV2 model.
3. The app instantly displays:
   - The predicted vehicle class (e.g., `🚗 Car` or `🏍️ Motorcycle`).
   - The confidence score (e.g., `98.5%`).
   - A probability breakdown bar chart across all 4 vehicle types.

To stop the Streamlit app when you're finished, press `Ctrl + C` in your PowerShell terminal.

---

## 📊 Understanding the Evaluation Metrics

When you train your model, `train.py` calculates four standard machine learning metrics:

| Metric | Plain English Meaning | Formula |
| :--- | :--- | :--- |
| **Accuracy** | Fraction of all images correctly classified. | $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$ |
| **Precision** | When the model predicts "Truck", how often is it actually a truck? | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ |
| **Recall** | Of all real trucks in the dataset, what fraction did the model identify? | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ |
| **F1-Score** | Balanced score combining precision and recall. Ideal when classes might be imbalanced. | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |

The **Confusion Matrix** shows a 4x4 table where diagonal entries represent correct predictions and off-diagonal entries represent misclassifications (e.g., a Truck mistaken for a Bus).

---

## 🛠️ Windows Troubleshooting Tips

1. **`python` command not recognized**:
   If Windows says `python is not recognized`, try using `py` instead:
   ```powershell
   py train.py
   py -m streamlit run app.py
   ```
2. **Out of Memory (OOM)**:
   If your computer slows down during training, decrease the batch size:
   ```powershell
   python train.py --batch-size 8
   ```
3. **Firewall alert when running Streamlit**:
   Windows Firewall may ask if Python can accept incoming connections. Allow access for private networks so your local browser can communicate with `localhost:8501`.
