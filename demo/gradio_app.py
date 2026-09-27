import gradio as gr
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vision_modal.inference import predict
from vision_modal.severity import estimate_severity
from vision_modal.gradcam import run_gradcam
from novel_modules.spread_simulation import run_ca
from novel_modules.treatment_simulator import simulate_treatment_impact
from novel_modules.lifespan_predictor import predict_lifespan
from rag_pipeline.rag_pipeline import get_treatment
import tensorflow as tf

MODEL_PATH = Path(__file__).parent.parent / "models" / "fytopod_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def run_pipeline(image, plant_age, stress_score, watering, light):
    if image is None:
        return "No image uploaded.", None, None

    # Save temp image
    import cv2, tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Predict
    result   = predict(tmp.name)
    disease  = result["disease_class"]
    conf     = result["confidence"]
    sev      = estimate_severity(conf)
    severity = sev["severity"]
    sev_score = sev["severity_score"]

    # Grad-CAM
    import matplotlib.pyplot as plt
    fig_gc, axes = plt.subplots(1, 3, figsize=(12, 4))
    heatmap, overlay = run_gradcam(model, tmp.name, result["pred_idx"])
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()

    # Treatment impact
    health = max(0, 100 - sev_score * 100 - stress_score * 0.3)
    sim = simulate_treatment_impact(health, sev_score, stress_score > 50)
    fig_tx, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sim["days"], sim["treated"],   color="#27ae60", label="With Treatment",    linewidth=2)
    ax.plot(sim["days"], sim["untreated"], color="#e74c3c", label="Without Treatment", linewidth=2, linestyle="--")
    ax.axhline(30, color="#f39c12", linestyle=":", label="Critical (30%)")
    ax.set_xlabel("Days"); ax.set_ylabel("Health Score (%)")
    ax.set_title("Treated vs Untreated Trajectory"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    # Treatment text
    treatment = get_treatment(disease, severity)
    lifespan  = predict_lifespan(sev_score, stress_score / 100,
                                  plant_age, watering, light)

    output_text = f"""
**Diagnosis:** {disease}
**Confidence:** {conf:.1%}
**Severity:** {severity}

**Treatment:**
{treatment['treatment']}

**Lifespan Estimate:**
Without treatment: {lifespan['days_without_treatment']} days
With treatment: {lifespan['days_with_treatment']} days

**Follow-up Questions:**
{treatment['followups']}
"""
    return output_text, fig_gc, fig_tx


demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Image(label="Upload Leaf Image", type="numpy"),
        gr.Slider(10, 365, value=60,  label="Plant Age (days)"),
        gr.Slider(0,  100, value=30,  label="Stress Score (0-100)"),
        gr.Slider(0,  1,   value=0.7, label="Watering Score (0-1)"),
        gr.Slider(0,  1,   value=0.7, label="Light Score (0-1)"),
    ],
    outputs=[
        gr.Markdown(label="Diagnosis and Treatment"),
        gr.Plot(label="Grad-CAM Explainability"),
        gr.Plot(label="Health Trajectory"),
    ],
    title="FytoPod — Plant Health Intelligence",
    description="Upload a leaf image to get disease diagnosis, Grad-CAM explanation, spread forecast, and treatment guidance.",
)

if __name__ == "__main__":
    demo.launch(share=True)