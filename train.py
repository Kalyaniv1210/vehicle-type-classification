"""
train.py
--------
Train a Vehicle Type Classifier using Transfer Learning with MobileNetV2.
Classes: Car, Bus, Truck, Motorcycle

Evaluates model performance using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Usage:
    python train.py --epochs 10 --batch-size 16
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Keep Keras cache locally inside workspace so model weights load offline
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")

# Silence oneDNN and TF logs to keep output beginner-friendly
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating plots safely
import matplotlib.pyplot as plt

import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = ["Bus", "Car", "Motorcycle", "Truck"]


def verify_dataset(train_path: Path, val_path: Path):
    """Verify that images exist in class subdirectories."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    missing = []
    total_train = 0
    total_val = 0

    for cls in EXPECTED_CLASSES:
        t_dir = train_path / cls
        v_dir = val_path / cls
        t_count = len([f for f in t_dir.iterdir() if f.suffix.lower() in valid_exts]) if t_dir.exists() else 0
        v_count = len([f for f in v_dir.iterdir() if f.suffix.lower() in valid_exts]) if v_dir.exists() else 0
        total_train += t_count
        total_val += v_count
        if t_count == 0 or v_count == 0:
            missing.append(cls)

    if total_train == 0 or total_val == 0:
        print("\n[ERROR] Dataset folders are empty or missing images!")
        print(f"  Training folder: {train_path}")
        print(f"  Validation folder: {val_path}")
        print("\nTip: To create a starter dataset to test this script immediately, run:")
        print("  python download_data.py --create-sample")
        sys.exit(1)

    print(f"[+] Dataset verified: {total_train} training images, {total_val} validation images.")


def load_datasets(train_path: Path, val_path: Path, test_path: Path, batch_size: int):
    """
    Load training, validation, and optional test datasets using Keras image_dataset_from_directory.
    Labels are inferred from folder names (Bus, Car, Motorcycle, Truck).
    """
    print("\n[1/5] Loading datasets from disk...")

    train_ds = keras.utils.image_dataset_from_directory(
        directory=str(train_path),
        labels="inferred",
        label_mode="int",
        class_names=EXPECTED_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = keras.utils.image_dataset_from_directory(
        directory=str(val_path),
        labels="inferred",
        label_mode="int",
        class_names=EXPECTED_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    test_ds = None
    # Check if a dedicated test split with images exists
    test_counts = [len(list((test_path / cls).glob("*.*"))) for cls in EXPECTED_CLASSES if (test_path / cls).exists()]
    if sum(test_counts) > 0:
        test_ds = keras.utils.image_dataset_from_directory(
            directory=str(test_path),
            labels="inferred",
            label_mode="int",
            class_names=EXPECTED_CLASSES,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            shuffle=False
        )
        print(f"  [+] Dedicated test dataset found: {sum(test_counts)} test images.")
    else:
        print("  [i] Dedicated data/test/ folder empty. Model evaluation will use validation dataset.")

    class_names = train_ds.class_names
    print(f"  Classes detected ({len(class_names)}): {class_names}")

    # Save class indices mapping for the Streamlit app
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open(MODELS_DIR / "class_indices.json", "w") as f:
        json.dump(class_mapping, f, indent=4)
    print(f"  Saved class index mapping to: {MODELS_DIR / 'class_indices.json'}")

    return train_ds, val_ds, test_ds, class_names


def build_transfer_learning_model(num_classes: int):
    """
    Constructs a MobileNetV2 Transfer Learning model:
    1. Input layer (224, 224, 3)
    2. Data augmentation (horizontal flip, subtle rotation & zoom)
    3. MobileNetV2 preprocessing (scales pixel values to [-1, 1])
    4. Pretrained MobileNetV2 base (ImageNet weights, frozen)
    5. Global Average Pooling + Dropout
    6. Dense Softmax Classification Head
    """
    print("\n[2/5] Building MobileNetV2 Transfer Learning Architecture...")

    # Data augmentation block (active only during training)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Base pretrained model
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze the base model to retain learned features from ImageNet
    base_model.trainable = False
    print("  Pretrained MobileNetV2 base loaded and frozen.")

    # Model input
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VehicleClassifier_MobileNetV2")
    return model


def plot_confusion_matrix(cm, class_names, output_path: Path):
    """Plots and saves a confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Vehicle Classification - Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix plot to: {output_path}")


def evaluate_model(model, eval_ds, class_names, split_name="Validation / Test"):
    """
    Evaluates the trained model on evaluation/test data:
    - Overall Accuracy
    - Macro Precision, Macro Recall, Macro F1-score
    - Weighted Precision, Recall, F1-score
    - Detailed per-class classification report
    - Confusion Matrix
    """
    print(f"\n[4/5] Evaluating model performance on {split_name} split...")

    y_true_list = []
    y_pred_list = []

    for images, labels in eval_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        y_pred_list.extend(pred_classes)
        y_true_list.extend(labels.numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Compute key metrics
    accuracy = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print("\n" + "=" * 65)
    print(f"       MODEL EVALUATION RESULTS ({split_name.upper()} DATASET)")
    print("=" * 65)
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {prec_macro * 100:.2f}%")
    print(f"Macro Recall:      {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print("-" * 65)
    print("Detailed Classification Report:\n")
    print(report_text)
    print("-" * 65)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 65)

    # Save metrics text report
    metrics_file = MODELS_DIR / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Vehicle Type Classification - Model Evaluation ({split_name})\n")
        f.write("=" * 65 + "\n")
        f.write(f"Overall Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Macro Precision:   {prec_macro * 100:.2f}%\n")
        f.write(f"Macro Recall:      {rec_macro * 100:.2f}%\n")
        f.write(f"Macro F1-Score:    {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted F1-Score: {f1_weighted * 100:.2f}%\n")
        f.write("-" * 65 + "\n\n")
        f.write(report_text + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"[+] Saved evaluation metrics to: {metrics_file}")

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, MODELS_DIR / "confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Train Vehicle Type Classifier with MobileNetV2")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    args = parser.parse_args()

    # Step 1: Verify data existence
    verify_dataset(TRAIN_DIR, VAL_DIR)

    # Step 2: Load data
    train_ds, val_ds, test_ds, class_names = load_datasets(TRAIN_DIR, VAL_DIR, TEST_DIR, args.batch_size)

    # Performance optimization: prefetch data batches
    train_ds = train_ds.prefetch(buffer_size=1)
    val_ds = val_ds.prefetch(buffer_size=1)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=1)

    # Step 3: Build model
    model = build_transfer_learning_model(num_classes=len(class_names))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Step 4: Train with callbacks
    best_model_path = MODELS_DIR / "vehicle_classifier_mobilenetv2.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\n[3/5] Starting model training for {args.epochs} epoch(s)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Ensure model is saved if early stopping or best checkpoint did not trigger
    if not best_model_path.exists():
        model.save(str(best_model_path))

    print(f"\n[+] Best trained model saved to: {best_model_path}")

    # Step 5: Evaluate model on separate test set (or val if test empty)
    eval_ds = test_ds if test_ds is not None else val_ds
    split_name = "Test" if test_ds is not None else "Validation"
    evaluate_model(model, eval_ds, class_names, split_name=split_name)

    print("\n[5/5] Training and evaluation complete!")
    print("To launch the web application, run:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
