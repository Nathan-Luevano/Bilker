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

class MarkdownExtractor:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    # Extract metadata and content from md files    
    def extract_markdown(self, md_path: Path) -> Dict[str, Any]:
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse md for structure
            md = markdown.Markdown(extensions=['toc', 'fenced_code'])
            html = md.convert(content)
            
            metadata = {
                'filename': md_path.name,
                'title': md_path.stem.replace('-', ' ').replace('_', ' ').title(),
                'doc_type': 'writeup' if 'writeup' in md_path.name.lower() else 'documentation'
            }
            
            return {
                'metadata': metadata,
                'content': content,
                'html': html
            }
        
        except Exception as e:
            logger.error(f"Error processing Markdown {md_path}: {str(e)}")
            return None

class CodeExtractor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.code_extensions = {'.py', '.c', '.cpp', '.h', '.sh', '.js', '.java', '.php', '.rb', '.go'}
    
    # Extract code files with context
    def extract_code(self, code_path: Path) -> Dict[str, Any]:
        try:
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            metadata = {
                'filename': code_path.name,
                'language': code_path.suffix[1:] if code_path.suffix else 'text',
                'title': f"Code: {code_path.name}",
                'doc_type': 'code'
            }
            
            # Context based on dir structure
            context_info = self._get_code_context(code_path)
            
            return {
                'metadata': metadata,
                'content': content,
                'context': context_info
            }
        
        except Exception as e:
            logger.error(f"Error processing code file {code_path}: {str(e)}")
            return None
    
    def _get_code_context(self, code_path: Path) -> str:
        parts = code_path.parts
        context_clues = []
        
        # Extract context from dir
        for part in parts:
            if any(keyword in part.lower() for keyword in ['ctf', 'challenge', 'exploit', 'pwn', 'crypto', 'web', 'reverse']):
                context_clues.append(part)
        
        # Look for README or documentation in same directory
        readme_files = list(code_path.parent.glob('README*')) + list(code_path.parent.glob('*.md'))
        if readme_files:
            context_clues.append(f"Documentation available: {readme_files[0].name}")
        
        return " | ".join(context_clues) if context_clues else "General code file"

class ImageExtractor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

    # Extract text from images using OCR
    def extract_image_text(self, image_path: Path) -> Dict[str, Any]:
        if not self.config.enable_ocr:
            return None
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            metadata = {
                'filename': image_path.name,
                'title': f"Image: {image_path.name}",
                'doc_type': 'image',
                'ocr_extracted': True
            }
            
            return {
                'metadata': metadata,
                'content': text.strip() if text.strip() else "No text detected in image",
                'image_path': str(image_path)
            }
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

class LLMFormatter:    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def format_to_qa(self, chunk: Dict[str, Any]) -> List[Dict[str, str]]:
        try:
            doc_type = chunk.get('doc_type', 'document')
            title = chunk.get('title', 'Unknown')
            content = chunk['text']
            
            prompt = self._create_formatting_prompt(content, doc_type, title)
            
            response = self._call_local_llm(prompt)
            qa_pairs = self._parse_llm_response(response)
            
            return qa_pairs
        
        except Exception as e:
            logger.error(f"Error formatting chunk: {str(e)}")
            return []
    
    def _create_formatting_prompt(self, content: str, doc_type: str, title: str) -> str:
        base_instruction = """
Convert the following content into high-quality question-answer pairs for training a CTF/cybersecurity AI assistant.

Focus on:
- Technical procedures and methodologies
- Tool usage and commands
- Vulnerability analysis and exploitation techniques
- Code explanations and security concepts
- Step-by-step problem-solving approaches

Format each Q&A pair as:
Q: [specific, actionable question]
A: [detailed, technical answer with examples where appropriate]

Ensure answers are comprehensive but concise, suitable for training an AI assistant."""

        type_specific = {
            'research_paper': "Focus on extracting methodologies, findings, and technical approaches from this research paper.",
            'writeup': "Extract step-by-step problem-solving procedures and techniques from this CTF writeup.",
            'code': "Explain the code functionality, security implications, and usage context.",
            'documentation': "Extract practical guidance, procedures, and reference information.",
            'image': "Format any technical information or procedures visible in this image content."
        }
        
        specific_instruction = type_specific.get(doc_type, "Extract relevant technical knowledge and procedures.")
        
        return f"""{base_instruction}

{specific_instruction}

Document: {title}
Content:
{content}

Generate Q&A pairs:"""
    # Formats content into Q&A using LLM
    def _call_local_llm(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.config.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return ""
        
        except Exception as e:
            logger.error(f"Error calling local LLM: {str(e)}")
            return ""
    
    # Parses LLM for current Q&A pairs
    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        qa_pairs = []
        lines = response.split('\n')
        
        current_q = ""
        current_a = ""
        in_answer = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Q:') or line.startswith('Question:'):
                # Save previous Q&A if exists
                if current_q and current_a:
                    qa_pairs.append({
                        'question': current_q.strip(),
                        'answer': current_a.strip()
                    })
                
                current_q = line.replace('Q:', '').replace('Question:', '').strip()
                current_a = ""
                in_answer = False
            
            elif line.startswith('A:') or line.startswith('Answer:'):
                current_a = line.replace('A:', '').replace('Answer:', '').strip()
                in_answer = True
            
            elif in_answer and line:
                current_a += " " + line
            
            elif not in_answer and current_q and line:
                current_q += " " + line
        
        # Save final Q&A pair
        if current_q and current_a:
            qa_pairs.append({
                'question': current_q.strip(),
                'answer': current_a.strip()
            })
        
        return qa_pairs