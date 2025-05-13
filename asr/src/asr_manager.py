
import os
import torch
import torchaudio
import tempfile
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class ASRManager:
    def __init__(self, model_path="C:/repos/til-25-here4food/asr/src/whisper-fine-tuned"):
        """
        Initialize the ASR Manager with the fine-tuned Whisper model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor and model
        try:
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to default Whisper Small model")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        
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
        """
        Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            # Load audio from the temporary file
            try:
                waveform, sample_rate = torchaudio.load(temp_audio_path)
            except Exception as audio_load_error:
                print(f"Error loading audio: {audio_load_error}")
                return ""
            
            # Remove the temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_audio_path}: {e}")
            
            # Preprocess audio
            audio_array = self.preprocess_audio(waveform, sample_rate)
            
            # Process audio with Whisper processor
            inputs = self.processor(
                audio_array,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error in ASR processing: {e}")
            return ""

# Test function
# def test_asr_manager():
#     """
#     Test the ASR Manager with a sample audio file
#     """
#     # Path to a test audio file
#     test_audio_path = "C:/repos/til-25-here4food/datasets/yolo_dataset/asr/sample_0.wav"
    
#     if not os.path.exists(test_audio_path):
#         print(f"Test audio file not found at {test_audio_path}")
#         return
    
#     # Load test audio
#     with open(test_audio_path, "rb") as f:
#         audio_bytes = f.read()
    
#     # Initialize ASR Manager
#     asr_manager = ASRManager()
    
#     # Test transcription
#     transcription = asr_manager.asr(audio_bytes)
    
#     print(f"Test audio transcription: {transcription}")


# if __name__ == "__main__":
#     test_asr_manager()