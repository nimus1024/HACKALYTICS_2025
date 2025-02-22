import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset as HFDataset, Audio
import numpy as np
import librosa

# Load the Whisper processor
processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='English', task='transcribe')

# Function to load transcripts from a file
def load_transcripts(file_path):
    with open(file_path, 'r') as f:
        transcripts = f.readlines()
    # Remove newline characters and strip whitespace
    transcripts = [line.strip() for line in transcripts if line.strip()]
    return transcripts

# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, whisper_transcripts, correct_transcripts):
        self.audio_files = audio_files
        self.whisper_transcripts = whisper_transcripts
        self.correct_transcripts = correct_transcripts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio file
        audio, sr = librosa.load(self.audio_files[idx], sr=16000)
        speech = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]

        # Tokenize correct transcript (ground truth)
        text = processor.tokenizer(self.correct_transcripts[idx], return_tensors="pt").input_ids[0]

        return {'speech': speech, 'text': text}

# Load transcripts
whisper_transcripts = load_transcripts('/Users/sabrinazhao/Documents/ai/HACKALYTICS_2025/all_transcripts.txt')
correct_transcripts = load_transcripts('/Users/sabrinazhao/Documents/ai/HACKALYTICS_2025/correct.txt')

# Ensure the number of transcripts matches
assert len(whisper_transcripts) == len(correct_transcripts), "Mismatch in the number of transcripts!"

# List of audio files (WAV format)
audio_files = [
    "/Users/sabrinazhao/Downloads/videos/cleft_21M_Baby_Gary.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_11M_Baby_Gary.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Baby_Gary.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Ben_sat-new.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_14M_Carly_cuddled.wav",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Cheeky_Charlie.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_08M_Elle_wanted.wav",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Funny_Sean.wav",
    "/Users/sabrinazhao/Downloads/videos/cleft_21M_Happy_Karen.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Happy_Karen-new.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_I_saw.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_16M_I_saw.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_20M_I_saw.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Jen_and-new.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Ken_likes.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_01F_Kenny_drank.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Kevin_got.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Kevin_got-new.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Kevin_got.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_Liz_played.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_My_daddy-new.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_My_granny.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_12M_My_granny.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_15M_my_granny.wav",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_Naughty_Neil-new.wav",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Naughty_Neil.wav"
    # Add paths to all your WAV files here
]

# Ensure the number of audio files matches the number of transcripts
assert len(audio_files) == len(whisper_transcripts), "Mismatch in the number of audio files and transcripts!"

# Create custom dataset
dataset = CustomDataset(audio_files, whisper_transcripts, correct_transcripts)

# Collate function for DataLoader
def collate_fn(data):
    speech = [{'input_features': i['speech']} for i in data]
    speech = processor.feature_extractor.pad(speech, return_tensors='pt').input_features

    text = [{'input_ids': i['text']} for i in data]
    text = processor.tokenizer.pad(text, return_tensors='pt').input_ids

    return {'speech': speech, 'text': text}

# Create DataLoader
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, drop_last=True)

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    def forward(self, speech, text):
        outputs = self.model(input_features=speech, labels=text)
        return outputs.logits

# Initialize model
model = Model()

# Training function
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)

    for epoch in range(1):  # Adjust the number of epochs as needed
        for i, data in enumerate(loader):
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(**data)

            loss = loss_fn(out.flatten(end_dim=-2), data['text'].flatten()) / 4
            loss.backward()
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    model.to('cpu')

# Train the model
train()