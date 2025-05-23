import io
import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
import os 

class ASRManager:
    def __init__(self, model_dir="whisper-ct2"):  
        self.frequency = 16000
        self.model = WhisperModel(
            model_dir,
            device="cuda",
            compute_type="float16",
            local_files_only=True
        )
        # Create 5 seconds of silence at 16kHz
        dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
        # Warm-up: forces model to load + do first-time setup
        self.asr(dummy_audio)
    
    def asr(self, audio_bytes: bytes) -> str:
        # Handle different input types
        if isinstance(audio_bytes, bytes):
            # Convert bytes to numpy array
            audio_file = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_file, dtype=np.float32)
            
            # Resample if necessary
            if sample_rate != self.frequency:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.frequency)
        else:
            # Assume it's already a numpy array
            audio_data = audio_bytes
        
        # Run transcription - Fixed: audio_data not audio*data, proper unpacking
        segments, info = self.model.transcribe(audio_data, beam_size=5, language="en")
        output = ""
        for segment in segments:
            output += segment.text
            
        return output