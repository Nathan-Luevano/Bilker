echo "Setting up bilker env"

# Conda check
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create conda environment
conda create -n bilker python=3.11 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bilker

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Check for tesseract (OCR)
if ! command -v tesseract &> /dev/null; then
    echo "!Tesseract OCR not found!"
    echo "To enable image text extraction, install with:"
    echo "sudo apt-get install tesseract-ocr"
    echo "Or set ENABLE_OCR = False in config.py"
fi

# Ollama 
if ! command -v ollama &> /dev/null; then
    echo "Ollama installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama already installed."
fi

# Just using llama3.1:8b for now no specific model needed
echo "Checking for llama3.1:8b model..."
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Downloading llama3.1:8b"
    ollama pull llama3.1:8b
fi

echo "Environment setup complete!"
echo "To activate: conda activate bilker"
echo "To process data: python process_data.py"