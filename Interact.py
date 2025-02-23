
from transformers import pipeline
import gradio as gr

# Specify the task explicitly
pipe = pipeline("automatic-speech-recognition", model="Lun798/model1")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small",
    description="Realtime demo for English speech recognition using a fine-tuned Whisper small model.",
)

iface.launch(debug=True)
