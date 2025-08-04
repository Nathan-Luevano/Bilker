import sys
import requests
from pathlib import Path

def check_dependencies():
    print("Checking Bilker environment...")
    
    # Check Python packages
    required_packages = ['PyPDF2', 'requests', 'PIL', 'markdown']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    print("\nChecking Ollama setup...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama server running")
            
            # Check if llama3.1:8b is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if any('llama3.1:8b' in name for name in model_names):
                return True
            else:
                print("[ERROR] llama3.1:8b model not found")
                print("Run: ollama pull llama3.1:8b")
                return False
        else:
            print("[ERROR] Ollama server not responding")
            return False
    
    except requests.exceptions.RequestException:
        print("[ERROR] Cannot connect to Ollama")
        print("Start Ollama with: ollama serve")
        return False

# Check if data directory exists and has files
def check_data_directory():
    print("\nChecking data directory...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("[ERROR] Data directory not found")
        print("Looking for data in: ./data")
        return False
    
    files = list(data_dir.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    
    if file_count == 0:
        print("[ERROR] No files found in data directory")
        return False
    
    print(f"[OK] Found {file_count} files in data directory")
    return True

def show_data_overview():
    print("\nData Overview:")
    
    data_dir = Path("data")
    file_types = {}
    
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    for ext, count in sorted(file_types.items()):
        print(f"   {ext or 'no extension'}: {count} files")

def estimate_processing_time():
    data_dir = Path("data")
    file_count = len([f for f in data_dir.rglob("*") if f.is_file()])
    
    # Estimating 16s per file
    estimated_minutes = (file_count * 16) / 60
    
    print(f"\nEstimated processing time: {estimated_minutes:.1f} minutes")
    print("   (Actual time depends on file sizes and LLM response speed)")

def main():
    print("Bilker CTF Assistant - Quick Start")
    print("-"*50)
    
    # Check environment
    if not check_dependencies():
        print("\n[ERROR] Environment check failed. Please fix dependencies first.")
        sys.exit(1)
    
    if not check_ollama():
        print("\n[ERROR] Ollama check failed. Please fix Ollama setup first.")
        sys.exit(1)
    
    if not check_data_directory():
        print("\n[ERROR] Data directory check failed. Please add source files to ./data")
        sys.exit(1)
    
    show_data_overview()
    estimate_processing_time()
    
    print("\n" + "-"*50)
    print("[OK] All checks passed! Ready to process data.")
    
    confirm = input("\nStart data processing? [y/N]: ").lower().strip()
    
    if confirm in ['y', 'yes']:
        print("\nStarting data processing...")
        print("Monitor progress with: tail -f bilker_processing.log")
        print("-" * 50)
        
        # Import and run main processor
        try:
            from process_data import main as process_main
            process_main()
        except ImportError:
            print("Error: process_data.py not found. Please ensure all files are in place.")
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Error during processing: {str(e)}")
    else:
        print("\nProcessing cancelled. Run again when ready.")

if __name__ == "__main__":
    main()