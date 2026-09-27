import tensorflow as tf
import numpy as np
import cv2

LAST_CONV_LAYER = "Conv_1"  # MobileNetV2 last conv layer

def get_gradcam_heatmap(model, img_array: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Generates Grad-CAM heatmap for the predicted class.
    img_array: shape (1, 224, 224, 3), already normalised
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_score = predictions[:, pred_idx]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img_path: str, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlays Grad-CAM heatmap on original image.
    Returns BGR image array.
    """
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def run_gradcam(model, image_path: str, pred_idx: int, save_path: str = None):
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    heatmap = get_gradcam_heatmap(model, img_array, pred_idx)
    overlay = overlay_heatmap(image_path, heatmap)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return heatmap, overlay_rgb