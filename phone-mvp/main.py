import torch
import ollama
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import os
from gtts import gTTS # For Text-to-Speech
import time

# --- Configuration and Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

print(f"Loading Whisper model '{model_id}' to {device}...")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please ensure you have the necessary Hugging Face models downloaded and sufficient GPU memory if using CUDA.")
    exit() # Exit if models can't be loaded

# --- Audio Recording Functions ---
def record_audio_segment(filename, duration=5, samplerate=16000):
    """Records a segment of audio and saves it to a file."""
    print(f"Recording for {duration} seconds... Please speak now.")
    temp_wav = "temp_recording_segment.wav" # Use a unique temp file name

    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait() # Wait for the recording to finish

        wav.write(temp_wav, samplerate, audio) # Save as WAV first
        audio_segment = AudioSegment.from_wav(temp_wav)
        audio_segment.export(filename, format="mp3") # Convert to MP3
        print(f"Audio segment saved to {filename}")
        return True
    except Exception as e:
        print(f"Error during audio recording: {e}")
        return False
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav) # Clean up temporary WAV file

def transcribe_audio(file_path):
    """Transcribes an audio file using the loaded Whisper pipeline."""
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found at {file_path}")
        return ""
    print(f"Transcribing audio from {file_path}...")
    try:
        result = whisper_pipe(file_path, return_timestamps=False) # return_timestamps=False for cleaner text
        transcribed_text = result["text"].strip()
        print(f"Transcribed: \"{transcribed_text}\"")
        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

# --- Text-to-Speech Function ---
def speak(text, lang='en'):
    """Converts text to speech and plays it."""
    if not text:
        return
    print(f"Ollama speaking: \"{text}\"")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts_audio_file = "ollama_response.mp3"
        tts.save(tts_audio_file)
        # Play the audio using pydub and sounddevice
        audio = AudioSegment.from_mp3(tts_audio_file)
        # Convert to numpy array for sounddevice
        data = np.array(audio.get_array_of_samples())
        # Play at the correct sample rate and channels
        sd.play(data, samplerate=audio.frame_rate, channels=audio.channels)
        sd.wait() # Wait for playback to finish
        os.remove(tts_audio_file) # Clean up the TTS audio file
    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        print("Please ensure you have an active internet connection for gTTS.")

# --- Voice Chat with History ---
def voice_chat_with_history(ollama_model="llama3.2", record_duration=5):
    """
    Initiates a voice chat session with the Ollama model, maintaining conversation history.
    """
    messages = [] # This list will store the chat history

    print("\n--- Welcome to Voice Chat with Ollama! ---")
    print(f"Using Ollama model: {ollama_model}")
    print("Each turn, you'll speak for approximately {} seconds.".format(record_duration))
    print("Say 'exit' or 'goodbye' to end the conversation.")
    print("------------------------------------------")

    while True:
        user_audio_file = "user_input_audio.mp3"
        # Record user's audio
        if not record_audio_segment(user_audio_file, duration=record_duration):
            continue # Skip this turn if recording failed

        # Transcribe user's audio
        user_text = transcribe_audio(user_audio_file)
        os.remove(user_audio_file) # Clean up user audio file

        if not user_text:
            print("Could not transcribe audio. Please try speaking more clearly.")
            speak("I didn't catch that. Could you please repeat?")
            continue

        # Check for exit command
        if user_text.lower() in ['exit', 'goodbye', 'bye']:
            speak("Exiting chat. Goodbye!")
            break

        # Add user's message to the history
        messages.append({"role": "user", "content": user_text})

        try:
            # Send the entire conversation history to Ollama
            print(f"Sending to Ollama ({ollama_model})...")
            response = ollama.chat(model=ollama_model, messages=messages)

            # Extract the model's response
            model_response = response["message"]["content"].strip()
            print(f"Ollama's raw response: {model_response}")

            # Add the model's response to the history
            messages.append({"role": "assistant", "content": model_response})

            # Speak Ollama's response
            speak(model_response)

        except Exception as e:
            print(f"An error occurred during Ollama interaction: {e}")
            print("Please ensure Ollama server is running and the model is available.")
            speak("I'm sorry, I encountered an error. Please check my connection.")
            # If an error occurs, remove the last user message to avoid
            # sending it again on the next iteration without a response.
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            time.sleep(2) # Wait a bit before next turn

# --- Main Execution ---
if __name__ == "__main__":
    # Example usage:
    voice_chat_with_history(ollama_model="llama3.2", record_duration=5)
