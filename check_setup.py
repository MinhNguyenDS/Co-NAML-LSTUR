#!/usr/bin/env python3
"""
Setup verification script for Co-NAML-LSTUR project.
Run this script to verify that your environment is properly configured.
"""

import sys
import importlib
import torch
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_pytorch():
    """Check PyTorch installation and device availability"""
    try:
        print(f"✅ PyTorch {torch.__version__} - OK")
        
        # Check device availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - {gpu_count} GPU(s): {gpu_name}")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  CPU only - GPU acceleration not available")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'transformers', 
        'tensorboard', 'tqdm', 'nltk'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_data_structure():
    """Check if data directories exist"""
    data_dirs = ['data', 'data/train', 'data/val', 'data/test']
    checkpoint_dir = 'checkpoint'
    
    print("\n📁 Data Structure Check:")
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ - exists")
        else:
            print(f"⚠️  {dir_path}/ - missing (run data preprocessing)")
    
    if os.path.exists(checkpoint_dir):
        checkpoints = list(Path(checkpoint_dir).rglob("*.pth"))
        if checkpoints:
            print(f"✅ checkpoint/ - {len(checkpoints)} checkpoint(s) found")
        else:
            print("⚠️  checkpoint/ - no checkpoints found (need to train model)")
    else:
        print("⚠️  checkpoint/ - missing (will be created during training)")

def check_model_config():
    """Check if model configurations are accessible"""
    try:
        from config import Co_NAML_LSTURConfig, NAML_LSTUR_DKNConfig
        print("✅ Model configurations - OK")
        
        # Test config instantiation
        config = Co_NAML_LSTURConfig()
        print(f"✅ Co_NAML_LSTUR config - batch_size: {config.batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ Model configuration error: {e}")
        return False

def check_transformers():
    """Check if transformers and DistilBERT are working"""
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        
        # Test tokenizer loading
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        print("✅ DistilBERT tokenizer - OK")
        
        # Test model loading (just check if it can be instantiated)
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        print("✅ DistilBERT model - OK")
        
        return True
    except Exception as e:
        print(f"❌ Transformers error: {e}")
        return False

def main():
    """Run all setup checks"""
    print("🔍 Co-NAML-LSTUR Setup Verification")
    print("=" * 40)
    
    checks = []
    
    # Basic checks
    print("\n📦 Environment Check:")
    checks.append(check_python_version())
    checks.append(check_pytorch())
    
    # Package checks
    print("\n📚 Package Check:")
    packages_ok, missing = check_required_packages()
    checks.append(packages_ok)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
    
    # Model checks
    print("\n🤖 Model Check:")
    checks.append(check_model_config())
    checks.append(check_transformers())
    
    # Data structure check
    check_data_structure()
    
    # Summary
    print("\n" + "=" * 40)
    if all(checks):
        print("🎉 All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Prepare your data: python data_preprocess.py")
        print("2. Train the model: python train.py --model_name Co_NAML_LSTUR")
        print("3. Run inference: python inference.py --model_name Co_NAML_LSTUR")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Update PyTorch: pip install torch --upgrade")
        print("- Check Python version: python --version")
    
    print("\n📖 For detailed instructions, see README.md")

if __name__ == "__main__":
    main() 