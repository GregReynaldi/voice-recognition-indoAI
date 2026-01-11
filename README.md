# Voice Recognition with Transformer Models

A comprehensive voice recognition system that implements and compares state-of-the-art transformer models for speech recognition, including Whisper and Wav2Vec2. The project uses the MINDS-14 dataset to evaluate model performance across multiple languages.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to develop and evaluate a voice recognition system using cutting-edge transformer models. The system is designed to:

- Transcribe speech in multiple languages
- Evaluate model performance using Word Error Rate (WER) and cosine similarity
- Compare different transformer architectures for speech recognition
- Fine-tune models for improved performance

The project provides a comprehensive analysis of how different models perform on both English and non-English speech data, offering insights into the strengths and limitations of each approach.

## Dataset

The project uses the **MINDS-14** dataset from Hugging Face, which contains:

- Speech data in 14 different languages
- Banking-related intents (e.g., balance inquiries, card blocking)
- Transcriptions for evaluation
- Audio samples at 16kHz sampling rate

Languages included: Czech, German, English, Spanish, French, Italian, Korean, Dutch, Polish, Portuguese, Russian, and Chinese.

## Models Implemented

### 1. Whisper
- **Model**: `openai/whisper-large-v3`
- **Type**: Multilingual speech recognition model
- **Key Features**: Automatic language detection, strong performance across languages

### 2. Wav2Vec2
- **Model**: `facebook/wav2vec2-base-100h`
- **Type**: Self-supervised speech representation model
- **Key Features**: Pre-trained on 100 hours of speech data

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended for model training and inference)

### Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Required Packages
- numpy
- pandas
- matplotlib
- torch
- librosa
- datasets
- transformers
- evaluate
- sentence_transformers

## Usage

### 1. Data Preparation
The dataset is automatically downloaded from Hugging Face when running the notebook:

```python
minds_14 = datasets.load_dataset("PolyAI/minds14", "all")
minds_14 = minds_14.cast_column("audio", Audio(sampling_rate=16000))
```

### 2. Exploratory Data Analysis
Run the EDA section to visualize:
- Language distribution
- Intent class distribution
- Audio waveforms
- Mel-spectrograms

### 3. Model Evaluation
Evaluate the performance of different models:

```python
# Example: Whisper model evaluation
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Generate predictions
results = test_dataset.map(func, batched=True, batch_size=32)
```

## Results

### Performance Comparison

| Model | Dataset | WER | Cosine Similarity |
|-------|---------|-----|-------------------|
| Whisper | English | 0.18 | 0.95 |
| Whisper | Non-English | 0.36 | 0.98 |
| Whisper | Overall | 0.41 | 0.96 |
| Wav2Vec2 | English | 0.61 | 0.70 |

### Key Insights
- **Whisper** demonstrates the best overall performance, especially on English data
- All models show high cosine similarity scores, indicating good semantic alignment
- Performance varies significantly between English and non-English data
- Fine-tuning has the potential to further improve model performance

## Technical Details

### Project Structure
```
├── main.ipynb          # Main notebook with complete implementation
├── config.py           # Utility functions for audio processing
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

### Config.py Functions
The `config.py` file contains utility functions for audio processing:

| Function | Description | Parameters | Return Value |
|----------|-------------|------------|--------------|
| `get_feature()` | Extracts audio features from dataset | `minds_14` (Dataset), `index` (int) | `audio` (array), `sr` (int), `label` (str), `translate` (str) |
| `generateAudio2Text()` | Generates text from audio using a model | `minds_14` (Dataset), `index` (int), `processor`, `model` | `answer` (list), `label` (str), `translate` (str) |
| `cleaning_text()` | Cleans and normalizes text | `sentence` (str) | Cleaned `sentence` (str) |

### Audio Processing
- **Sampling Rate**: 16kHz
- **Feature Extraction**: Mel-spectrograms
- **Preprocessing**: Audio normalization and padding

### Model Evaluation
- **Metric**: Word Error Rate (WER)
- **Semantic Evaluation**: Cosine similarity between predicted and reference embeddings
- **Batched Inference**: For improved performance

### Hardware Requirements
- **Training**: GPU with at least 16GB VRAM recommended
- **Inference**: Can run on CPU, but GPU recommended for faster results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing access to pre-trained models and datasets
- [PolyAI](https://polyai.com/) for creating the MINDS-14 dataset
- [OpenAI](https://openai.com/) for developing the Whisper model
- [Facebook AI Research](https://ai.facebook.com/) for developing Wav2Vec2

## Contact

- GitHub: [@GregReynaldi](https://github.com/GregReynaldi)
- Email: [gregoriusreynaldi@gmail.com](mailto:gregoriusreynaldi@gmail.com)

---

*Built with ❤️ for speech recognition research*
