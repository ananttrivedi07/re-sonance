# tts_engine.py
import io
import wave
import numpy as np
import sounddevice as sd
from piper import PiperVoice

class LocalTTSEngine:
    def __init__(self, model_path="en_US-lessac-medium.onnx", config_path="en_US-lessac-medium.onnx.json"):
        print("Initializing local Piper TTS Engine...")
        try:
            self.voice = PiperVoice.load(model_path, config_path=config_path)
            print("Piper TTS loaded successfully.")
        except Exception as e:
            print(f"Could not load Piper voice models: {e}")
            self.voice = None

    def speak(self, text: str):
        """Converts text string into raw audio directly hitting speakers via memory buffers."""
        if not text.strip() or not self.voice:
            return
        
        try:
            wav_buffer = io.BytesIO()
            # Open the buffer as a formal write-only wave structure
            with wave.open(wav_buffer, "wb") as wav_file:
                # Piper writes the structural headers (channels, sample width, frame rate)
                self.voice.synthesize_wav(text, wav_file)
            
            # Rewind and read the formatted audio back out
            wav_buffer.seek(0)
            with wave.open(wav_buffer, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                num_frames = wav_file.getnframes()
                
                audio_bytes = wav_file.readframes(num_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # FIX: Reshape the array to (frames, channels) so sounddevice auto-detects it.
                # This removes the need to pass 'channels=' as a keyword argument entirely!
                audio_data = audio_data.reshape(-1, num_channels)
                
                # Play the reshaped data safely
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"TTS Streaming Playback Error: {e}")