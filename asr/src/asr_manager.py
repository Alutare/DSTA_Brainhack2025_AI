
# import os
# import torch
# import torchaudio
# import tempfile
# import io 
# import soundfile as sf
# import numpy as np
# from torch.jit import script
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch.nn.functional as F


# class ASRManager:
#     def __init__(self, model_path="til-25-here4food/asr/src/whisper-fine-tuned"):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         # Load with optimization flags
#         self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
#         self.model = WhisperForConditionalGeneration.from_pretrained(
#             model_path, 
#             local_files_only=True,
#             torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,  # Use FP16 on GPU
#             low_cpu_mem_usage=True
#         ).to(self.device)
        
#         # Enable optimizations
#         self.model.eval()
#         if self.device == 'cuda':
#             self.model = torch.compile(self.model, mode="reduce-overhead")  # PyTorch 2.0+
        
#     def asr(self, audio_bytes: bytes) -> str:
#         # More efficient audio loading
#         audio_file = io.BytesIO(audio_bytes)
#         audio_data, sample_rate = sf.read(audio_file, dtype=np.float32)

#         # Resample only if necessary
#         target_sr = self.processor.feature_extractor.sampling_rate
#         if sample_rate != target_sr:
#             audio_data = torchaudio.functional.resample(
#                 torch.from_numpy(audio_data), sample_rate, target_sr
#             ).numpy()

#         # Process features more efficiently
#         inputs = self.processor.feature_extractor(
#             audio_data, 
#             sampling_rate=target_sr, 
#             return_tensors="pt",
#             padding=True
#         )

#         input_features = inputs.input_features.to(self.device)

#         # Optimized inference
#         with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=='cuda'):
#             generated_ids = self.model.generate(
#                 input_features,
#                 max_length=448,  # Set reasonable max length
#                 num_beams=1,     # Use greedy decoding instead of beam search
#                 do_sample=False,
#                 use_cache=True,
#                 pad_token_id=self.processor.tokenizer.pad_token_id
#             )

#         transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         return transcription

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

# def test_asr_manager():
#     """
#     Test the ASR Manager with a sample audio file
#     """
#     # Path to a test audio file
#     test_audio_path = "advanced/asr/sample_1.wav" 
#     if not os.path.exists(test_audio_path):
#         print(f"Test audio file not found at {test_audio_path}")
#         return
    
#     try:
#         # Initialize ASR Manager
#         print("Initializing ASR Manager...")
#         asr_manager = ASRManager()
#         print("ASR Manager initialized successfully!")
        
#         # Load as bytes and test transcription
#         print("Testing transcription...")
#         with open(test_audio_path, "rb") as f:
#             audio_bytes = f.read()
#         transcription = asr_manager.asr(audio_bytes)  # Now works with bytes directly
#         print(f"Test audio transcription: {transcription}")
        
#     except Exception as e:
#         print(f"Error during testing: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     test_asr_manager()