#!/usr/bin/env python3
"""
Simple script to download tokenizer.json for your whisper-ct2 model
"""

import os
import shutil
import json

def download_tokenizer(model_dir="whisper-ct2"):
    """Download tokenizer.json from Hugging Face"""
    
    print(f"Downloading tokenizer.json to {model_dir}...")
    
    try:
        # Method 1: Using huggingface_hub
        from huggingface_hub import hf_hub_download
        
        print("1. Downloading tokenizer.json from openai/whisper-small...")
        file_path = hf_hub_download(
            repo_id="openai/whisper-small",
            filename="tokenizer.json",
            local_files_only=False  # Allow download
        )
        
        # Copy to your model directory
        dest_path = os.path.join(model_dir, "tokenizer.json")
        shutil.copy2(file_path, dest_path)
        
        print(f"✅ tokenizer.json downloaded to: {dest_path}")
        
        # Verify file exists and has content
        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            print(f"   File size: {size:,} bytes")
            
            # Quick validation - check if it's valid JSON
            with open(dest_path, 'r') as f:
                data = json.load(f)
                if 'model' in data and 'vocab' in data:
                    print(f"   ✅ File appears to be valid tokenizer")
                else:
                    print(f"   ⚠️  File may not be a valid tokenizer")
        
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub"])
        return download_tokenizer(model_dir)  # Try again
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def convert_vocabulary(model_dir="whisper-ct2"):
    """Convert vocabulary.json to vocabulary.txt if needed"""
    
    vocab_json_path = os.path.join(model_dir, "vocabulary.json")
    vocab_txt_path = os.path.join(model_dir, "vocabulary.txt")
    
    if os.path.exists(vocab_json_path) and not os.path.exists(vocab_txt_path):
        print(f"\n2. Converting vocabulary.json to vocabulary.txt...")
        
        try:
            with open(vocab_json_path, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
            
            # Sort by token ID and write to .txt file
            sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
            
            with open(vocab_txt_path, 'w', encoding='utf-8') as f:
                for token, _ in sorted_vocab:
                    f.write(f"{token}\n")
            
            print(f"✅ Created vocabulary.txt with {len(sorted_vocab)} tokens")
            return True
            
        except Exception as e:
            print(f"❌ Vocabulary conversion failed: {e}")
            return False
    else:
        if os.path.exists(vocab_txt_path):
            print(f"✅ vocabulary.txt already exists")
        else:
            print(f"⚠️  No vocabulary.json found to convert")
        return True

def main():
    """Main function"""
    
    # Update this path to match your model directory
    MODEL_DIR = "whisper-ct2"
    
    print("Tokenizer Download Script")
    print("=" * 30)
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  📁 {item}")
        return
    
    print(f"Model directory: {MODEL_DIR}")
    print(f"Current files:")
    for file in os.listdir(MODEL_DIR):
        print(f"  📄 {file}")
    
    # Download tokenizer.json
    success1 = download_tokenizer(MODEL_DIR)
    
    # Convert vocabulary if needed
    success2 = convert_vocabulary(MODEL_DIR)
    
    if success1 and success2:
        print(f"\n🎉 SUCCESS!")
        print(f"Your model directory now has all required files.")
        print(f"\nTry running your ASRManager again:")
        print(f"  ASRManager(model_dir='{MODEL_DIR}')")
    else:
        print(f"\n❌ Some operations failed. Check the errors above.")

if __name__ == "__main__":
    main()