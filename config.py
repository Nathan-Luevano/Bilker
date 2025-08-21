from pathlib import Path
from dataclasses import dataclass

@dataclass
class BilkerConfig:
    # Project settings
    PROJECT_NAME = "bilker"
    VERSION = "1.0.0"
    
    # Data processing
    MAX_CHUNK_SIZE = 4000
    OVERLAP_SIZE = 200
    LOCAL_MODEL = "deepseek-r1:32b"

    # Paths
    DATA_DIR = Path("data")
    PROCESSED_DIR = Path("processed")
    MODELS_DIR = Path("models")
    
    # Processing options
    ENABLE_OCR = True            # Only use if pytesseract is installed
    ENABLE_CODE_EXTRACTION = True
    SKIP_EXISTING = True         
    
    # Future training settings
    BASE_MODEL = "qwen3-coder:30B"
    TRAINING_BATCH_SIZE = 1
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 4096