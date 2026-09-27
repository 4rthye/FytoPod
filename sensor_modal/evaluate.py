import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              recall_score)
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "results" / "sensor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_val, y_val, model_name="LSTM"):
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    recall = recall_score(y_val, y_pred)
    report = classification_report(y_val, y_pred,
                                   target_names=["No Stress", "Stress"])
    print(f"\n{model_name} Results")
    print(f"Stress Recall: {recall:.2%}")
    print(report)
    return y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, model_name="LSTM", save=True):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["No Stress", "Stress"],
                yticklabels=["No Stress", "Stress"],
                cmap="Blues")
    plt.title(f"FytoPod — {model_name} Confusion Matrix", fontweight="bold")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"confusion_matrix_{model_name.lower()}.png"
        plt.savefig(path, dpi=150)
        print(f"Saved to {path}")
    plt.show()


def plot_training_history(history, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train", color="#27ae60", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val",   color="#3498db", linewidth=2)
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", color="#e74c3c", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val",   color="#e67e22", linewidth=2)
    axes[1].set_title("Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("FytoPod LSTM — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "training_history.png"
        plt.savefig(path, dpi=150)
        print(f"Saved to {path}")
    plt.show()