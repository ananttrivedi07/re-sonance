import torch
import ollama
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import os
from gtts import gTTS
import time
import webrtcvad

# --- Configuration and Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

# VAD parameters
VAD_SAMPLE_RATE = 16000 # Must be 8000, 16000, 32000, or 48000
VAD_FRAME_DURATION_MS = 30 # Must be 10, 20, or 30 ms
VAD_PADDING_MS = 300 # Add padding to detected speech segments
VAD_AGGRESSIVENESS = 3 # 0 (least aggressive) to 3 (most aggressive)

# Calculate frame size in samples
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

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


# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# --- Audio Recording and VAD Functions ---
def record_and_detect_speech(filename, samplerate=VAD_SAMPLE_RATE):
    """
    Listens continuously, detects speech using VAD, and saves the detected speech
    segment to a file.
    """
    print("\nListening for speech... (Say 'exit' or 'goodbye' to end)")
    audio_buffer = []
    in_speech_segment = False
    speech_start_time = None
    
    # For padding at the end of speech
    ring_buffer = []
    ring_buffer_size = int(VAD_PADDING_MS / VAD_FRAME_DURATION_MS)

    # Open an audio input stream
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            # Read a frame of audio
            frame, overflowed = stream.read(VAD_FRAME_SIZE)
            if overflowed:
                print("Audio input buffer overflowed!")

            # Flatten frame and convert to bytes for VAD
            frame_flattened = frame.flatten()
            frame_bytes = frame_flattened.tobytes()

            # Check if it's a speech frame
            is_speech = vad.is_speech(frame_bytes, samplerate)

            if not in_speech_segment:
                # Not currently in a speech segment, looking for start
                if is_speech:
                    print("Speech detected! Recording...")
                    in_speech_segment = True
                    speech_start_time = time.time()
                    # Add flattened data from ring buffer
                    for buf in ring_buffer:
                        audio_buffer.extend(buf)
                    audio_buffer.extend(frame_flattened)
                    ring_buffer = []
                else:
                    # Keep non-speech frames in ring buffer for potential pre-padding
                    ring_buffer.append(frame_flattened)
                    if len(ring_buffer) > ring_buffer_size:
                        ring_buffer.pop(0)
            else:
                # Currently in a speech segment
                audio_buffer.extend(frame_flattened)
                if not is_speech:
                    # Speech ended, start padding counter
                    ring_buffer.append(frame_flattened)
                    if len(ring_buffer) >= ring_buffer_size:
                        # Enough non-speech frames for padding, segment ended
                        print("Speech ended. Processing...")
                        break
                else:
                    # Speech continues, clear ring buffer
                    ring_buffer = []

    if not audio_buffer:
        print("No speech detected in this turn.")
        return False

    try:
        # Convert buffered audio to numpy array
        audio_data = np.array(audio_buffer, dtype=np.int16)
        
        # Save temporary WAV file
        temp_wav = "temp_speech_segment.wav"
        wav.write(temp_wav, samplerate, audio_data)
        audio_segment = AudioSegment.from_wav(temp_wav)
        audio_segment.export(filename, format="mp3")
        print(f"Speech segment saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving audio segment: {e}")
        return False
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

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
def voice_chat_with_history(ollama_model="llama3.2"):
    """
    Initiates a voice chat session with the Ollama model, maintaining conversation history.
    Uses VAD for continuous listening.
    """
    messages = [] # This list will store the chat history

    print("\n--- Welcome to Continuous Voice Chat with Ollama! ---")
    print(f"Using Ollama model: {ollama_model}")
    print("Speak when you are ready. The system will detect your speech.")
    print("Say 'exit' or 'goodbye' to end the conversation.")
    print("------------------------------------------")

    while True:
        user_audio_file = "user_input_vad_audio.mp3"
        # Record user's audio using VAD
        if not record_and_detect_speech(user_audio_file):
            continue # If no speech was detected, continue listening

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
    voice_chat_with_history(ollama_model="llama3.2")
