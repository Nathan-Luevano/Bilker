# Bilker

![til](./Bilker_GIF.gif)

**CTF/Cybersecurity AI Assistant using QLoRA Fine-tuning**

Transforms diverse cybersecurity educational content (research papers, CTF writeups, exploit code) into training datasets for specialized AI assistant development.

## Overview

**Architecture**: Two-phase modular pipeline for reliable data processing and model training
**Base Model**: QWen3-Coder:30B fine-tuned with QLoRA
**Dataset**: 12,601 Q&A pairs from 2,241 cybersecurity source files

## Pipeline

```
Raw Data → process_data.py → Chunks → chunk_processor.py → Training Dataset → QLoRA Training
```

**Phase 1**: Content extraction and intelligent chunking
**Phase 2**: LLM-powered Q&A pair generation using local llama3.1:8b

## Requirements

**Data Processing**: 8GB+ RAM, local LLM capability
**Model Training**: NVIDIA GPU 16GB+ VRAM, 32GB+ system RAM

## Usage

```bash
conda create -n bilker python=3.11
pip install -r requirements.txt
ollama pull llama3.1:8b

python process_data.py      # Extract and chunk data
python chunk_processor.py   # Generate Q&A pairs
```

## Results

From comprehensive cybersecurity dataset including PicoCTF challenges, HackTheBox writeups, academic research, and exploit repositories:
- 87% chunk-to-QA conversion success rate
- Professional-grade training dataset ready for fine-tuning
- Modular architecture enables resumable processing and experimentation

## Status

- **Complete**: Data processing pipeline, training dataset generation
- **In Progress**: QLoRA fine-tuning implementation
- **Planned**: Model deployment and evaluation

Built for cybersecurity education and ethical security research.
