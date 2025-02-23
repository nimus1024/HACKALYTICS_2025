from datasets import Dataset, DatasetDict, Audio

# Your JSON data as a string (replace this with loading from a file if needed)
data = {
    "disordered_child_speech_sentences": [
      {
        "stimulus": "Baby Gary got a bag of lego",
        "audio": "/Users/sabrinazhao/Downloads/videos/cleft_21M_Baby_Gary.wav"
      }, 
      {
        "stimulus": "Baby Gary got a bag of lego",
        "audio": "/Users/sabrinazhao/Downloads/videos/Ultraphonix_11M_Baby_Gary.wav`"
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
    "path": audio_paths
})

# Cast the 'path' column to Audio type
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

# Split the dataset into train, validation, and test sets
train_testval = dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test+val
test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)  # 50% for both test and validation

# Create a DatasetDict
common_voice = DatasetDict({
    "train": train_testval["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
})

# Print the DatasetDict
print(common_voice)
