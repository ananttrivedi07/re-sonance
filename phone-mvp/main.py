import torch
import ollama
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import time
import webrtcvad
import pyttsx3

# --- Configuration and Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

VAD_SAMPLE_RATE = 16000 
VAD_FRAME_DURATION_MS = 30 
VAD_PADDING_MS = 300 
VAD_AGGRESSIVENESS = 3 
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
    exit()

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# --- In-Memory Audio Recording ---
def record_and_detect_speech(samplerate=VAD_SAMPLE_RATE):
    print("\nListening... (Say 'exit' or 'goodbye' to end)")
    audio_buffer = []
    in_speech_segment = False
    ring_buffer = []
    ring_buffer_size = int(VAD_PADDING_MS / VAD_FRAME_DURATION_MS)

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            frame, overflowed = stream.read(VAD_FRAME_SIZE)
            frame_flattened = frame.flatten()
            is_speech = vad.is_speech(frame_flattened.tobytes(), samplerate)

            if not in_speech_segment:
                if is_speech:
                    print("Speech detected...")
                    in_speech_segment = True
                    for buf in ring_buffer:
                        audio_buffer.extend(buf)
                    audio_buffer.extend(frame_flattened)
                    ring_buffer = []
                else:
                    ring_buffer.append(frame_flattened)
                    if len(ring_buffer) > ring_buffer_size:
                        ring_buffer.pop(0)
            else:
                audio_buffer.extend(frame_flattened)
                if not is_speech:
                    ring_buffer.append(frame_flattened)
                    if len(ring_buffer) >= ring_buffer_size:
                        break
                else:
                    ring_buffer = []

    if not audio_buffer:
        return None

    return np.array(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

def transcribe_audio_memory(audio_ndarray):
    if audio_ndarray is None:
        return ""
    try:
        result = whisper_pipe(audio_ndarray, return_timestamps=False)
        return result["text"].strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

# --- Fresh Engine Local TTS (Fixes the "Second Response" Bug) ---
def speak_sentence(text):
    """
    Spaks a single sentence by spinning up a brief, self-contained execution context.
    This bypasses pyttsx3's native loop lockups across multiple chat iterations.
    """
    if not text.strip():
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(text.strip())
        engine.runAndWait()
        # Explicit lifecycle cleanup
        del engine
    except Exception as e:
        print(f"\nTTS Error: {e}")

# --- Streaming Voice Chat with History ---
def voice_chat_with_history(ollama_model="llama3.2"):
    messages = []
    print("\n--- Streaming Local Voice Chat Connected ---")

    while True:
        user_audio = record_and_detect_speech()
        if user_audio is None:
            continue

        user_text = transcribe_audio_memory(user_audio)
        print(f"\nYou: {user_text}")

        if not user_text:
            speak_sentence("I didn't catch that. Could you please repeat?")
            continue

        if user_text.lower().strip().rstrip('.') in ['exit', 'goodbye', 'bye']:
            speak_sentence("Goodbye!")
            break

        messages.append({"role": "user", "content": user_text})

        try:
            print(f"Ollama stream starting...", end="", flush=True)
            
            # Request token streaming from Ollama
            response_stream = ollama.chat(model=ollama_model, messages=messages, stream=True)

            full_assistant_response = ""
            sentence_buffer = ""
            
            print("\nAI: ", end="", flush=True)

            for chunk in response_stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)  # Print token to console immediately
                
                full_assistant_response += token
                sentence_buffer += token

                # If we encounter a punctuation boundary, speak it instantly
                if any(p in token for p in ['.', '?', '!', '\n']):
                    clean_sentence = sentence_buffer.strip().replace('\n', ' ')
                    if clean_sentence:
                        speak_sentence(clean_sentence)
                    sentence_buffer = "" # Reset buffer for next sentence

            # Speak any trailing words left over without punctuation at the end
            if sentence_buffer.strip():
                speak_sentence(sentence_buffer.strip())

            # Save the full response string to history
            messages.append({"role": "assistant", "content": full_assistant_response})
            print() # Newline after response completes

        except Exception as e:
            print(f"\nAn error occurred during Ollama interaction: {e}")
            speak_sentence("I encountered an error. Please try again.")
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            time.sleep(1)

if __name__ == "__main__":
    voice_chat_with_history(ollama_model="llama3.2")