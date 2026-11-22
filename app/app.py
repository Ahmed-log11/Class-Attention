from collections import Counter

import gradio as gr
import spaces
from ultralytics import YOLO

model = YOLO("best.pt")


@spaces.GPU
def analyze(image, conf):
    if image is None:
        return None, "Upload a photo or take one with your webcam."

    # Gradio gives RGB, YOLO expects BGR for numpy arrays
    result = model(image[..., ::-1], conf=conf, verbose=False)[0]
    annotated = result.plot()[..., ::-1]  # back to RGB for display

    labels = [result.names[int(c)] for c in result.boxes.cls]
    counts = Counter(labels)

    absent = counts.get("empty_seat", 0)
    present = len(labels) - absent
    focused = counts.get("focused", 0)

    if present == 0:
        return annotated, "No students detected. Try a clearer photo or lower the confidence."

    score = focused / present * 100
    lines = [f"## Class attention: {score:.0f}%", "",
             f"Students present: {present}  |  Empty seats: {absent}", ""]
    for cls in ["focused", "looking_away", "phone", "sleeping"]:
        lines.append(f"- {cls}: {counts.get(cls, 0)}")
    return annotated, "\n".join(lines)


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="numpy", sources=["upload", "webcam"], label="Classroom photo"),
        gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detections"),
        gr.Markdown(),
    ],
    title="Classroom Attention Detection",
    description=(
        "Upload a classroom photo or use your webcam. The model labels each student as "
        "focused, looking away, on their phone, or sleeping, and gives an overall attention score. "
        "Photos are processed only to show the result and are not stored."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
