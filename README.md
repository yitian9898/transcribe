# Whisper Transcribe

This script uses OpenAI's Whisper API to transcribe audio files. If the audio file is larger than 25MB, it will be split into smaller chunks, transcribed, and the results combined.

## Requirements

- Python 3.8+
- OpenAI API key
- ffmpeg (system dependency)

## Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install ffmpeg (Mac):**

```bash
brew install ffmpeg
```

If you don't have Homebrew, install it from [https://brew.sh/](https://brew.sh/).

3. **Set your OpenAI API key:**

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

```bash
python whisper_transcribe.py path/to/audiofile -o output.txt
```

- Replace `path/to/audiofile` with your audio file path.
- The `-o` flag specifies the output text file (default: `transcript.txt`). 