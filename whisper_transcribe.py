import os
import sys
import math
import argparse
import tempfile
import subprocess
import openai
from tqdm import tqdm

# Set your OpenAI API key here or via environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

CHUNK_SIZE_MB = 25
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

def get_file_size(path):
    return os.path.getsize(path)

def split_audio(input_path, chunk_size_bytes):
    """
    Splits the audio file into chunks of <= chunk_size_bytes using ffmpeg.
    Returns a list of chunk file paths.
    """
    total_size = get_file_size(input_path)
    if total_size <= chunk_size_bytes:
        return [input_path]

    # Get duration of the audio file
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())

    # Estimate number of chunks
    num_chunks = math.ceil(total_size / chunk_size_bytes)
    chunk_duration = duration / num_chunks

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_duration
        output_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_path)[1])
        output_chunk.close()
        chunk_paths.append(output_chunk.name)
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start),
            '-t', str(chunk_duration),
            '-c', 'copy',
            output_chunk.name
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return chunk_paths

def transcribe_audio(file_path, model="whisper-1"):
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    return transcript.text

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI Whisper API.")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output text file")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("Error: Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    input_path = args.input
    output_path = args.output

    print(f"Checking file size...")
    file_size = get_file_size(input_path)
    if file_size <= CHUNK_SIZE_BYTES:
        print("File is within size limit. Transcribing directly...")
        text = transcribe_audio(input_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcription saved to {output_path}")
    else:
        print(f"File is larger than {CHUNK_SIZE_MB}MB. Splitting into chunks...")
        chunk_paths = split_audio(input_path, CHUNK_SIZE_BYTES)
        print(f"Transcribing {len(chunk_paths)} chunks...")
        all_text = []
        for chunk_path in tqdm(chunk_paths, desc="Transcribing chunks"):
            text = transcribe_audio(chunk_path)
            all_text.append(text)
            os.remove(chunk_path)  # Clean up chunk file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        print(f"Combined transcription saved to {output_path}")

if __name__ == "__main__":
    main()