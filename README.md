# Hacklytics 2025: Personalized Speech-to-Text for Children with Down Syndrome

## Project Description

This application, developed during Hacklytics 2025, is a Personalized Speech-to-Text Generator tailored for people with speech disabilities, with a specific focus on children with Down syndrome.

**Problem:** Standard speech recognition tools often struggle with the unique speech patterns of children with Down syndrome, leading to inaccurate transcriptions.

**Solution:** This project leverages generative AI to create a lightweight model that dynamically adapts to individual speech characteristics. By learning and refining its accuracy based on real-time user feedback, the application aims to provide a more reliable transcription experience.

**Goal:** To overcome the challenges of delayed speech development, pronunciation difficulties, and complex sentence formation commonly experienced by children with Down syndrome.

This project aims to develop a personalized speech-to-text generator specifically designed to accurately transcribe the speech of children with Down syndrome. Recognizing that standard speech recognition tools often struggle with non-standard speech patterns, this application leverages generative AI to dynamically adapt and learn a user's unique speech characteristics over time.

We are addressing the challenge of delayed speech development, pronunciation difficulties, and complex sentence formation common in children with Down syndrome, by creating a tool that improves transcription accuracy through real-time learning and user feedback.

## Features

* Voice Recording: A user-friendly microphone interface for capturing speech input.
* Transcription: Immediate text transcription with incremental learning capabilities.
* Personalized Adaptation: The AI model learns and adapts to individual speech patterns over time, improving accuracy.
* Personalization: A short voice training period enables the system to rapidly create a personalized speech model.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nimus1024/HACKALYTICS_2025.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install transformers
    pip install gradio
    pip install ffmpeg (make sure ffprobe is in your path)
    ```
3.  **Run the application:**
    ```bash
    Open Interact.py and run.
    ```

## How It Works

The application utilizes a pre-trained speech-to-text model as a foundation. A generative AI layer is then implemented to personalize the transcription process. This layer continuously refines its accuracy based on user input and corrections. The user can train the system with a brief voice sample, creating a personalized model that adapts to their unique speech patterns.

## Target Audience

This tool is specifically designed for children with Down syndrome and their caregivers/strangers, providing a more reliable and accurate speech-to-text solution.

## Future Improvements

* Expanding the training data to include a wider range of speech patterns.
* Implementing more advanced generative AI techniques for improved personalization.
* Developing a mobile application for greater accessibility.
* Implementing different languages.
* Improving the user interface based on user feedback.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to submit a pull request or open an issue.

## License

MIT License

## Acknowledgments

* [Acknowledge any relevant libraries, APIs, or resources used]
* Hacklytics 2025 (hosted by Data Science @ GT)
