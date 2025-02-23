from datasets import Dataset, DatasetDict, Audio
import os

# Your JSON data as a string (replace this with loading from a file if needed)
data = {
    "disordered_child_speech_sentences": [
        {
            "stimulus": "Baby Gary got a bag of lego",
            "audio": "/Users/sabrinazhao/Downloads/videos/cleft_21M_Baby_Gary.wav"
        }, 
        {
            "stimulus": "Baby Gary got a bag of lego",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_11M_Baby_Gary.wav"
        },
        {
            "stimulus": "Baby Gary's got a bag of lego",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Baby_Gary.wav"
        },
        {
            "stimulus": "Ben sat on the pin",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Ben_sat-new.wav"
        },
        {
            "stimulus": "Carly cuddled her dolly",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_14M_Carly_cuddled.wav"
        },
        {
            "stimulus": "Cheeky Charlie's watching a football match",
            "audio": "/Users/sabrinazhao/Downloads/videos/cleft_16M_Cheeky_Charlie.wav"
        },
        {
            "stimulus": "Elle wanted to sell ten hens to Ken",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_08M_Elle_wanted.wav"
        },
        {
            "stimulus": "Funny Sean was washing a dirty dish",
            "audio": "/Users/sabrinazhao/Downloads/videos/cleft_16M_Funny_Sean.wav"
        },
        {
            "stimulus": "Happy Karen is making a cake",
            "audio": "/Users/sabrinazhao/Downloads/videos/cleft_21M_Happy_Karen.wav"
        },
        {
            "stimulus": "Happy Karen is making a cake",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Happy_Karen-new.wav"
        },
        {
            "stimulus": "I saw Sam sitting on a bus",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_I_saw.wav"
        },
        {
            "stimulus": "I saw Sam sitting on a bus",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_16M_I_saw.wav"
        },
        {
            "stimulus": "I saw Sam sitting on a bus",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_20M_I_saw.wav"
        },
        {
            "stimulus": "Jen and Jan were drinking gin",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_05M_Jen_and-new.wav"
        },
        {
            "stimulus": "Ken likes scones with cream and apricot jam",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Ken_likes.wav"
        },
        {
            "stimulus": "Kenny drank a tiny tin of coke",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_01F_Kenny_drank.wav"
        },
        {
            "stimulus": "Kevin got a cab to the coast",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_Kevin_got.wav"
        },
        {
            "stimulus": "Kevin got a cab to the coast",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_13M_Kevin_got-new.wav"
        },
        {
            "stimulus": "Kevin got a cab to the coast",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_17M_Kevin_got.wav"
        },
        {
            "stimulus": "Liz played with the toys and was amused",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_10M_Liz_played.wav"
        },
        {
            "stimulus": "My daddy mended a door",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_My_daddy-new.wav"
        },
        {
            "stimulus": "My granny Maggie got a golden gown",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_04M_My_granny.wav"
        },
        {
            "stimulus": "My granny Maggie got a golden gown",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_12M_My_granny.wav"
        },
        {
            "stimulus": "My granny Maggie got a golden gown",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_15M_my_granny.wav"
        },
        {
            "stimulus": "Naughty Neil saw a robin in a nest",
            "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_09M_Naughty_Neil-new.wav"
        },
        {
            "stimulus": "Naughty Neil saw a robin in a nest",
            "audio": "/Users/sabrinazhao/Downloads/videos/cleft_16M_Naughty_Neil.wav"
        }
    ]
}

# Extract the list of sentences and audio paths
sentences = [item["stimulus"] for item in data["disordered_child_speech_sentences"]]
audio_paths = [item["audio"] for item in data["disordered_child_speech_sentences"]]

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({
    "sentence": sentences,
    "audio": audio_paths  # Use "audio" as the key instead of "path"
})

# Cast the 'audio' column to Audio type
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split the dataset into train, validation, and test sets
train_testval = dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test+val
test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)  # 50% for both test and validation

# Create a DatasetDict
common_voice = DatasetDict({
    "train": train_testval["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
})

# Print the first example in the training set
print(common_voice["train"][0])

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")


tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input: {input_str}")
print(f"Decoded w/ special: {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal: {input_str == decoded_str}")

print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset, 
    remove_columns=common_voice["train"].column_names,  # Correctly access column names
    num_proc=4
)
import torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.mps.empty_cache() 
#torch.tensor([1,2,3], device="mps")

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract the already-computed input features
        input_features = [feature["input_features"] for feature in features]
        # Pad input features using the feature extractor's pad method
        batch = self.processor.feature_extractor.pad(
            [{"input_features": inp} for inp in input_features], return_tensors="pt"
        )
        
        # Extract the already-computed labels
        labels = [feature["labels"] for feature in features]
        # Pad the labels using the tokenizer's pad method
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": label} for label in labels], return_tensors="pt"
        )
        # Replace padding token ids with -100 for loss computation
        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels_tensor

        return batch

    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=4,  
    learning_rate=1e-5,
    warmup_steps=20,
    max_steps=100,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,  
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

processor.save_pretrained(training_args.output_dir)

from transformers import Seq2SeqTrainer

torch.cuda.amp
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()






