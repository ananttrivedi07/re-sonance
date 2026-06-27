import torch
import numpy as np
import sounddevice as sd
import webrtcvad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class AudioSTTEngine:
    def __init__(self, model_id="openai/whisper-large-v3-turbo", aggressiveness=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Audio & VAD Settings
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.padding_ms = 300
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self._init_whisper(model_id)

    def _init_whisper(self, model_id):
        print(f"Initializing Whisper '{model_id}' on {self.device}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("Whisper ready.")

    def listen_and_transcribe(self) -> str:
        """Listens to microphone, captures phrase via VAD, and returns text string."""
        print("\nListening for speech...")
        audio_buffer = []
        in_speech_segment = False
        ring_buffer = []
        ring_buffer_size = int(self.padding_ms / self.frame_duration_ms)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
            while True:
                frame, overflowed = stream.read(self.frame_size)
                frame_flattened = frame.flatten()
                is_speech = self.vad.is_speech(frame_flattened.tobytes(), self.sample_rate)

                if not in_speech_segment:
                    if is_speech:
                        print("Speech detected! Recording...")
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
                            print("Speech ended. Processing...")
                            break
                    else:
                        ring_buffer = []

        if not audio_buffer:
            return ""

        # Convert int16 buffer directly to normalized float32 numpy array
        audio_ndarray = np.array(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        try:
            result = self.whisper_pipe(audio_ndarray, return_timestamps=False)
            return result["text"].strip()
        except Exception as e:
            print(f"STT Error: {e}")
            return ""