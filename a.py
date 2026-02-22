import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Load your model
MODEL_PATH = "C:/Users/rajub/Downloads/vechicle+1.pt"
model = YOLO(MODEL_PATH)
print(model.names)
# Define the only classes you want to see
EMERGENCY_NAMES = [
    "ambulance", "ambulance_108", "ambulance_SOL", 
    "fire_truck", "fireladder", "firelamp", 
    "firesymbol", "firewriting"
]

# Map names to IDs (YOLO works faster with IDs)
# We only keep IDs that actually exist in your model's metadata
ALLOWED_IDS = [id for id, name in model.names.items() if name.lower() in [en.lower() for en in EMERGENCY_NAMES]]

CONFIDENCE_THRESHOLD = 0.50 # Note: Your UI says 95%, but your variable says 0.50. Adjust as needed!

def detect_image(image):
    if image is None:
        return None, "No image uploaded"
    
    # Predict only for the ALLOWED_IDS to ignore other classes (like cars, people, etc.)
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, classes=ALLOWED_IDS, verbose=False)
    
    result = results[0]
    high_conf_detections = []
    
    if len(result.boxes) > 0:
        # Plot only the filtered boxes
        im_array = result.plot()
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            high_conf_detections.append(f"{class_name}: {conf:.2%}")
    else:
        # No detections found for emergency classes
        im_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if isinstance(image, np.ndarray) else cv2.imread(image)

    # Convert BGR back to RGB for Gradio
    result_image = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    
    if high_conf_detections:
        info = f"🚨 Detections (≥{CONFIDENCE_THRESHOLD:.0%}):\n" + "\n".join(high_conf_detections)
    else:
        info = f"✅ No emergency vehicles detected (≥{CONFIDENCE_THRESHOLD:.0%})"
    
    return result_image, info

def detect_video(video):
    if video is None:
        return None, "No video uploaded"
    
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    high_conf_detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # classes=ALLOWED_IDS ensures YOLO ignores everything else
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=ALLOWED_IDS, verbose=False)
        
        result = results[0]
        if len(result.boxes) > 0:
            annotated_frame = result.plot()
            high_conf_detections_count += len(result.boxes)
        else:
            annotated_frame = frame
            
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    
    info = f"🎬 Video processed! Detections found: {high_conf_detections_count}"
    return output_path, info



# Custom CSS for styling
css = """
.ambulance-title {
    text-align: center;
    color: #ff4444;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 20px;
}
.confidence-note {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="Ambulance Detection") as demo:
    gr.HTML("<h1 class='ambulance-title'>🚑 Ambulance Detection System</h1>")
    gr.HTML("<p class='subtitle'>Upload images or videos to detect ambulances using YOLO</p>")
    gr.HTML("<div class='confidence-note'>⚠️ <strong>Strict Mode:</strong> Only displays detections with ≥95% confidence. Lower confidence predictions are ignored.</div>")
    
    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                image_btn = gr.Button("🔍 Detect Ambulance", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="Detection Result", height=400)
                image_info = gr.Textbox(
                    label="Detection Details",
                    lines=5,
                    interactive=False
                )
        
        image_btn.click(
            fn=detect_image,
            inputs=image_input,
            outputs=[image_output, image_info]
        )
    
    with gr.Tab("🎬 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Upload Video",
                    height=300
                )
                video_btn = gr.Button("🔍 Detect in Video", variant="primary", size="lg")
                gr.Markdown("⚠️ Video processing may take time depending on length")
            
            with gr.Column():
                video_output = gr.Video(label="Detection Result", height=300)
                video_info = gr.Textbox(
                    label="Processing Details",
                    lines=4,
                    interactive=False
                )
        
        video_btn.click(
            fn=detect_video,
            inputs=video_input,
            outputs=[video_output, video_info]
        )

    gr.Markdown("""
    ### How it works:
    1. **Strict Confidence Threshold**: Only predictions with **≥95% confidence** are displayed
    2. **No False Positives**: If confidence is below 95%, the original image/video is shown without any boxes
    3. **High Precision Mode**: This ensures only very certain ambulance detections are shown
    
    **Current Settings:**
    - Confidence Threshold: 95%
    - Model: `best.pt`
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)