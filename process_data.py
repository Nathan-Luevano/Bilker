import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
from datetime import datetime

import PyPDF2
import requests
from bs4 import BeautifulSoup
import markdown
from PIL import Image
import pytesseract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bilker_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    max_chunk_size: int = 4000  
    overlap_size: int = 200     

    local_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    
    data_dir: Path = Path("data")
    processed_dir: Path = Path("processed")
    chunks_dir: Path = Path("processed/chunks")
    formatted_dir: Path = Path("processed/formatted")
    metadata_dir: Path = Path("processed/metadata")
    
    enable_ocr: bool = True
    enable_code_extraction: bool = True
    skip_existing: bool = True

class DocumentChunker:
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def chunk_text(self, text: str, title: str = "", doc_type: str = "") -> List[Dict[str, Any]]:
        chunks = []
        words = text.split()
        
        if len(words) <= self.config.max_chunk_size:
            return [{
                'text': text,
                'chunk_id': 0,
                'title': title,
                'doc_type': doc_type,
                'total_chunks': 1
            }]
        
        chunk_size = self.config.max_chunk_size
        overlap = self.config.overlap_size
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'title': title,
                'doc_type': doc_type,
                'total_chunks': -1,  
                'overlap_start': i > 0,
                'overlap_end': i + chunk_size < len(words)
            })
        
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        return chunks

class PDFExtractor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                text_content = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append({
                            'page': page_num + 1,
                            'text': page_text.strip()
                        })
                
                metadata = {
                    'filename': pdf_path.name,
                    'num_pages': len(reader.pages),
                    'title': getattr(reader.metadata, 'title', pdf_path.stem) if reader.metadata else pdf_path.stem,
                    'doc_type': 'research_paper' if 'pdf' in pdf_path.suffix.lower() else 'document'
                }
                
                full_text = '\n\n'.join([page['text'] for page in text_content])
                
                return {
                    'metadata': metadata,
                    'content': full_text,
                    'pages': text_content
                }
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None

