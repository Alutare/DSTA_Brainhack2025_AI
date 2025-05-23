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
from sklearn.model_selection import KFold

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

def compute_metrics(pred, processor):
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

############ MODEL TRAINING WITH CV ############

def train_fold(train_dataset, val_dataset, fold_num, processor, output_dir="./whisper-cv"):
    """
    Train a single fold of the cross-validation.
    """
    # Initialize model for this fold
    model_name = "openai/whisper-small"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure model
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    fold_output_dir = f"{output_dir}/fold_{fold_num}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=fold_output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="epoch",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="wer",
        greater_is_better=False,
        num_train_epochs=5,
        load_best_model_at_end=True,
    )
    

    # Initialize trainer with processor for metrics
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        tokenizer=processor.feature_extractor,
    )
    
    # Train model
    result = trainer.train()
    
    # Get best validation WER
    eval_result = trainer.evaluate()
    
    # Save best model for this fold
    model.save_pretrained(fold_output_dir)
    processor.save_pretrained(fold_output_dir)
    
    return eval_result["eval_wer"], model

def train_whisper_with_cv(dataset, n_folds=5, output_dir="./whisper-cv"):
    """
    Train Whisper model using k-fold cross validation.
    """
    # Initialize processor (same for all folds)
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    
    # Convert dataset to indices for splitting
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    
    # Initialize KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_scores = []
    best_model = None
    best_score = float('inf')
    
    print(f"Starting {n_folds}-fold cross validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Create train and validation datasets for this fold
        train_dataset = dataset.select(train_idx.tolist())
        val_dataset = dataset.select(val_idx.tolist())
        
        # Train model for this fold
        fold_wer, model = train_fold(
            train_dataset, val_dataset, fold + 1, processor, output_dir
        )
        
        fold_scores.append(fold_wer)
        print(f"Fold {fold + 1} WER: {fold_wer:.4f}")
        
        # Keep track of best model
        if fold_wer < best_score:
            best_score = fold_wer
            best_model = model
            best_fold = fold + 1
    
    # Calculate average performance
    mean_wer = np.mean(fold_scores)
    std_wer = np.std(fold_scores)
    
    print(f"\n{'='*50}")
    print(f"Cross Validation Results:")
    print(f"{'='*50}")
    print(f"Fold WERs: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"Mean WER: {mean_wer:.4f} ± {std_wer:.4f}")
    print(f"Best Fold: {best_fold} (WER: {best_score:.4f})")
    
    # Save best model
    best_model_dir = f"{output_dir}/best_model"
    best_model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    
    # Save CV results
    cv_results = {
        "fold_scores": fold_scores,
        "mean_wer": mean_wer,
        "std_wer": std_wer,
        "best_fold": best_fold,
        "best_wer": best_score
    }
    
    with open(f"{output_dir}/cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    return best_model, processor, cv_results

############ MAIN EXECUTION FOR TRAINING ############

def main():
    """
    Main function to train the model and set up the ASR manager.
    """
    # Path configurations
    json_path = "advanced/asr/asr.jsonl"  # Update with your JSON path
    audio_dir = "advanced/asr"  # Update with your audio directory
    output_dir = "./whisper-fine-tuned1"
    
    # Initialize processor for data preparation
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    dataset = prepare_dataset(json_path, audio_dir, processor)
    
    # Train with cross validation
    best_model, processor, results = train_whisper_with_cv(
        dataset, 
        n_folds=5, 
        output_dir="./whisper-cv-results"
    )

if __name__ == "__main__":
    main()