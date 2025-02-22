import os
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import wave

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Speech-to-Text Generator")
        
        self.recognizer = sr.Recognizer()
        
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.transcription = tk.StringVar()
        
        self.create_widgets()
        
    def create_widgets(self):
        self.record_button = tk.Button(self.root, text="Record", command=self.record_audio)
        self.record_button.pack(pady=10)
        
        self.transcription_label = tk.Label(self.root, textvariable=self.transcription, wraplength=400)
        self.transcription_label.pack(pady=10)
        
        self.correct_button = tk.Button(self.root, text="Correct Transcription", command=self.correct_transcription)
        self.correct_button.pack(pady=10)
        
    def record_audio(self):
        # Initialize values
        sampling_rate = 44100
        channels = 1
        output_folder = "recordings"
        audio_data = []

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Callback function for capturing audio
        def callback(indata, frames, time, status):
            if status:
                print("Error:", status)
            if len(indata) > 0:
                audio_data.extend(indata.copy())  # Append the incoming audio data

        # Start recording
        stream = sd.InputStream(callback=callback, channels=channels, samplerate=sampling_rate)
        stream.start()
        messagebox.showinfo("Recording", "Recording started. Press OK to stop.")
        stream.stop()
        stream.close()

        # Save the recorded audio
        audio_filename = os.path.join(output_folder, f"audio_{len(os.listdir(output_folder)) + 1}.wav")
        if len(audio_data) == 0:
            messagebox.showerror("Error", "No sound was captured.")
        else:
            audio_data_int16 = np.array(audio_data, dtype=np.float32)
            audio_data_int16 = np.clip(audio_data_int16 * 32767, -32768, 32767).astype(np.int16)
            with wave.open(audio_filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sampling_rate)
                wf.writeframes(audio_data_int16.tobytes())
            messagebox.showinfo("Saved", f"Audio saved to {audio_filename}")

            # Transcribe the audio
            self.transcribe_audio(audio_filename)
    
    def transcribe_audio(self, audio_filename):
        try:
            with sr.AudioFile(audio_filename) as source:
                audio = self.recognizer.record(source)
            audio_data = audio.get_wav_data()
            input_values = self.tokenizer(audio_data, return_tensors="pt").input_values
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.decode(predicted_ids[0])
            self.transcription.set(transcription)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def correct_transcription(self):
        corrected_text = self.transcription.get()
        messagebox.showinfo("Corrected Transcription", corrected_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()