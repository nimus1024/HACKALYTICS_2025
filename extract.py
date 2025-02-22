import os
import torch
import librosa
from moviepy.editor import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def extract_audio_from_mp4(mp4_file, audio_file):
    try:
        # Load the MP4 file
        video = VideoFileClip(mp4_file)

        # Extract the audio
        audio = video.audio

        # Save the audio as a WAV file
        audio.write_audiofile(audio_file, codec='pcm_s16le')

        # Close the video file
        video.close()

        print(f"Audio extracted and saved as: {audio_file}")
    except Exception as e:
        print(f"Failed to extract audio from {mp4_file}: {e}")

def transcribe_audio_with_whisper(audio_file, transcript_file):
    try:
        # Load model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)

        # Process audio and generate input features
        input_features = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features

        # Force the model to transcribe in English
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids  # Force English transcription
            )

        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Append the transcribed text to the single transcript file
        with open(transcript_file, "a") as f:
            f.write(f"Transcript for {audio_file}:\n")
            f.write(transcription + "\n\n")

        print(f"Transcription appended for: {audio_file}")
    except Exception as e:
        print(f"Failed to transcribe audio from {audio_file}: {e}")

# List of MP4 file paths
mp4_files = [
    "/Users/sabrinazhao/Downloads/videos/cleft_21M_Baby_Gary.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_11M_Baby_Gary.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Baby_Gary.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Ben_sat-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_14M_Carly_cuddled.mp4",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Cheeky_Charlie.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_08M_Elle_wanted.mp4",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Funny_Sean.mp4",
    "/Users/sabrinazhao/Downloads/videos/cleft_21M_Happy_Karen.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Happy_Karen-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_I_saw.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_16M_I_saw.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_20M_I_saw.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Jen_and-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Ken_likes.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_01F_Kenny_drank.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Kevin_got.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Kevin_got-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Kevin_got.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_Liz_played.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_My_daddy-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_My_granny.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_12M_My_granny.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_15M_my_granny.mp4",
    "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_Naughty_Neil-new.mp4",
    "/Users/sabrinazhao/Downloads/videos/cleft_16M_Naughty_Neil.mp4"
]

# Set the path for the single transcript file
transcript_file = "/Users/sabrinazhao/Documents/ai/transcripts/all_transcripts.txt"

# Ensure the directory exists
os.makedirs(os.path.dirname(transcript_file), exist_ok=True)

# Loop through each MP4 file and process it
for mp4_file in mp4_files:
    # Step 1: Extract audio from MP4
    audio_file = mp4_file.replace(".mp4", ".wav")  # Assuming you want to save as WAV
    extract_audio_from_mp4(mp4_file, audio_file)

    # Step 2: Transcribe the extracted audio with Whisper and append to a single transcript file
    transcribe_audio_with_whisper(audio_file, transcript_file)