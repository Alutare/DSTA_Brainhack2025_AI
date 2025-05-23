import io
import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
import os 

class ASRManager:
    def __init__(self, model_path="whisper-fine-tuned"):
        """
        Initialize the ASR Manager with the fine-tuned Whisper model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
    
        # Set model parameters for inference
        self.model.config.suppress_tokens = []
        self.model.generation_config.input_ids = self.model.generation_config.forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = None
        
        # Set up an audio resampler for 16kHz (Whisper's expected sample rate)
        self.target_sample_rate = 16000
        
    def preprocess_audio(self, waveform, sample_rate):
        """
        Preprocess audio by converting to mono and resampling to 16kHz
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Preprocessed waveform numpy array
        """
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to numpy array and flatten
        return waveform.squeeze().numpy()
        
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