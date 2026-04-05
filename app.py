import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.h5")

def predict(image):
    img = cv2.resize(image, (128,128))
    img = img / 255.0
    img = img.reshape(1,128,128,3)

    pred = model.predict(img)
    label = np.argmax(pred)

    if label == 1:
        return "✅ Person is wearing a mask"
    else:
        return "❌ Person is NOT wearing a mask"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="😷 Face Mask Detection"
)

demo.launch(share=True)
