import os
import json
import base64
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Any
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset, Audio
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Union

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

############ DATA PREPARATION FUNCTIONS ############

def prepare_dataset(json_path, audio_dir, processor):
    """
    Prepare dataset for training from JSON file and audio directory.
    """
    # Read JSON data
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Create a list of dictionaries with audio paths and transcripts
    dataset_dict = {
        "audio": [os.path.join(audio_dir, item["audio"]) for item in data],
        "transcript": [item["transcript"] for item in data],
        "key": [item["key"] for item in data]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Process audio
    dataset = dataset.cast_column("audio", Audio())
    
    def prepare_features(batch):
        audio = batch["audio"]
        # Process audio
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features
        
        # Process text
        batch["input_features"] = input_features.squeeze()
        batch["labels"] = processor(text=batch["transcript"]).input_ids
        return batch
    
    # Apply preprocessing
    dataset = dataset.map(prepare_features, remove_columns=["audio"])
    
    return dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input_features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch

def compute_metrics(pred):
    """
    Compute word error rate metrics.
    """
    wer_metric = evaluate.load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Convert ids to text
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

############ MODEL TRAINING ############

def train_whisper_model(dataset, output_dir="./whisper-fine-tuned"):
    """
    Fine-tune the Whisper Small model.
    """
    # Initialize Whisper processor and model
    global processor
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure model
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.2)
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        #max_steps=2000,
        gradient_checkpointing=True,
        fp16=True,
        # evaluation_strategy="epoch",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        #save_steps=1000,
        #eval_steps=1000,
        logging_steps=25,
        save_strategy="no",
        metric_for_best_model="wer",
        greater_is_better=False,
        num_train_epochs=2,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Train model
    trainer.train()
    
    # Save model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    return model, processor

############ ASR MANAGER IMPLEMENTATION ############

class ASRManager:
    def __init__(self, model_path="./whisper-fine-tuned"):
        """
        Initialize the ASR Manager with the fine-tuned Whisper model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        
    def asr(self, audio_bytes: bytes) -> str:
        """
        Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """
        try:
            # Save the bytes to a temporary file
            temp_file = "temp_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            
            # Load audio
            waveform, sample_rate = torchaudio.load(temp_file)
            
            # If stereo, convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Process audio
            input_features = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Clean up
            os.remove(temp_file)
            
            return transcription
        
        except Exception as e:
            print(f"Error in ASR processing: {e}")
            return ""

############ API HANDLER IMPLEMENTATION ############

def handle_asr_request(request_data):
    """
    Handle the ASR request as per the specified format.
    
    Args:
        request_data: JSON input following the specified format
        
    Returns:
        A dictionary with predictions
    """
    asr_manager = ASRManager()
    predictions = []
    
    for instance in request_data.get("instances", []):
        # Decode base64 audio
        audio_bytes = base64.b64decode(instance["b64"])
        
        # Process audio
        transcript = asr_manager.asr(audio_bytes)
        
        # Add to predictions
        predictions.append(transcript)
    
    return {"predictions": predictions}

############ MAIN EXECUTION FOR TRAINING ############

def main():
    """
    Main function to train the model and set up the ASR manager.
    """
    # Path configurations
    json_path = "advanced/asr/asr.jsonl"  # Update with your JSON path
    audio_dir = "advanced/asr"  # Update with your audio directory
    output_dir = "./whisper-fine-tuned"
    
    # Initialize processor for dataset preparation
    global processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # Prepare dataset
    dataset = prepare_dataset(json_path, audio_dir, processor)
    
    # Train model
    model, processor = train_whisper_model(dataset, output_dir)
    
    # Initialize ASR Manager
    asr_manager = ASRManager(output_dir)
    
    print("Model training complete and ASR Manager ready!")

if __name__ == "__main__":
    main()