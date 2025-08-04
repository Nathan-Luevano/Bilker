import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import PyPDF2
from bs4 import BeautifulSoup
import markdown
from PIL import Image
import pytesseract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bilker_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    max_chunk_size: int = 4000  
    overlap_size: int = 200     
    
    data_dir: Path = Path("data")
    chunks_dir: Path = Path("processed/chunks")
    metadata_dir: Path = Path("processed/metadata")
    
    enable_ocr: bool = True
    enable_code_extraction: bool = True
    skip_existing: bool = True

class DocumentChunker:
    def __init__(self, config: ExtractionConfig):
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
                'total_chunks': -1,  # Will be updated
                'overlap_start': i > 0,
                'overlap_end': i + chunk_size < len(words)
            })
        
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        return chunks

class PDFExtractor:
    def __init__(self, config: ExtractionConfig):
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
                    'file_path': str(pdf_path),
                    'num_pages': len(reader.pages),
                    'title': getattr(reader.metadata, 'title', pdf_path.stem) if reader.metadata else pdf_path.stem,
                    'doc_type': self._classify_pdf_type(pdf_path)
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
    
    def _classify_pdf_type(self, pdf_path: Path) -> str:
        path_parts = str(pdf_path).lower()
        
        if any(keyword in path_parts for keyword in ['writeup', 'solution', 'walkthrough']):
            return 'writeup'
        elif any(keyword in path_parts for keyword in ['research', 'paper', 'analysis', 'evaluation']):
            return 'research_paper'
        elif any(keyword in path_parts for keyword in ['challenge', 'ctf', 'competition']):
            return 'challenge'
        else:
            return 'document'

class MarkdownExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
    
    def extract_markdown(self, md_path: Path) -> Dict[str, Any]:
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            metadata = {
                'filename': md_path.name,
                'file_path': str(md_path),
                'title': md_path.stem.replace('-', ' ').replace('_', ' ').title(),
                'doc_type': self._classify_markdown_type(md_path)
            }
            
            return {
                'metadata': metadata,
                'content': content
            }
        
        except Exception as e:
            logger.error(f"Error processing Markdown {md_path}: {str(e)}")
            return None
    
    def _classify_markdown_type(self, md_path: Path) -> str:
        path_parts = str(md_path).lower()
        
        if any(keyword in path_parts for keyword in ['writeup', 'solution', 'walkthrough']):
            return 'writeup'
        elif any(keyword in path_parts for keyword in ['readme', 'documentation', 'guide']):
            return 'documentation'
        elif any(keyword in path_parts for keyword in ['cheat', 'reference', 'resource']):
            return 'reference'
        else:
            return 'documentation'

class CodeExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.code_extensions = {'.py', '.c', '.cpp', '.h', '.sh', '.js', '.java', '.php', '.rb', '.go', '.rs'}
    
    def extract_code(self, code_path: Path) -> Dict[str, Any]:
        try:
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            metadata = {
                'filename': code_path.name,
                'file_path': str(code_path),
                'language': code_path.suffix[1:] if code_path.suffix else 'text',
                'title': f"Code: {code_path.name}",
                'doc_type': 'code',
                'context': self._get_code_context(code_path)
            }
            
            return {
                'metadata': metadata,
                'content': content
            }
        
        except Exception as e:
            logger.error(f"Error processing code file {code_path}: {str(e)}")
            return None
    
    def _get_code_context(self, code_path: Path) -> str:
        parts = code_path.parts
        context_clues = []
        
        for part in parts:
            if any(keyword in part.lower() for keyword in ['ctf', 'challenge', 'exploit', 'pwn', 'crypto', 'web', 'reverse', 'forensics']):
                context_clues.append(part)
        
        readme_files = list(code_path.parent.glob('README*')) + list(code_path.parent.glob('*.md'))
        if readme_files:
            context_clues.append(f"Documentation: {readme_files[0].name}")
        
        return " | ".join(context_clues) if context_clues else "General code file"

class ImageExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    
    def extract_image_text(self, image_path: Path) -> Dict[str, Any]:
        if not self.config.enable_ocr:
            return None
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            metadata = {
                'filename': image_path.name,
                'file_path': str(image_path),
                'title': f"Image: {image_path.name}",
                'doc_type': 'image',
                'ocr_extracted': True
            }
            
            return {
                'metadata': metadata,
                'content': text.strip() if text.strip() else "No text detected in image"
            }
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

class DataExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.chunker = DocumentChunker(config)
        self.pdf_extractor = PDFExtractor(config)
        self.md_extractor = MarkdownExtractor(config)
        self.code_extractor = CodeExtractor(config)
        self.image_extractor = ImageExtractor(config)
        
        # Create directories
        self._setup_directories()
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'errors': 0,
            'skipped_existing': 0
        }
    
    def _setup_directories(self):
        for directory in [self.config.chunks_dir, self.config.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def extract_all_data(self) -> Dict[str, Any]:
        logger.info("Starting data extraction and chunking...")
        
        all_files = self._discover_files()
        logger.info(f"Discovered {len(all_files)} files to process")
        
        for file_path in all_files:
            try:
                self._process_single_file(file_path)
                self.stats['files_processed'] += 1
                
                if self.stats['files_processed'] % 100 == 0:
                    logger.info(f"Processed {self.stats['files_processed']} files...")
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                self.stats['errors'] += 1
        
        self._save_processing_summary()
        
        logger.info(f"Extraction complete! {self.stats}")
        return self.stats
    
    def _discover_files(self) -> List[Path]:
        files = []
        
        for root, dirs, filenames in os.walk(self.config.data_dir):
            for filename in filenames:
                file_path = Path(root) / filename
                
                if (filename.startswith('.') or 
                    filename.endswith('.log') or
                    file_path.suffix.lower() in {'.zip', '.tar', '.gz'}):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _process_single_file(self, file_path: Path):
        # Generate unique file ID
        file_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        
        # Skip if already processed
        if self.config.skip_existing:
            existing_file = self.config.chunks_dir / f"{file_id}_chunks.json"
            if existing_file.exists():
                logger.debug(f"Skipping {file_path} (already processed)")
                self.stats['skipped_existing'] += 1
                return
        
        logger.info(f"Processing: {file_path}")
        
        # Extract content based on file type
        extracted_data = self._extract_content(file_path)
        if not extracted_data:
            return
        
        # Chunk the content
        chunks = self.chunker.chunk_text(
            extracted_data['content'],
            extracted_data['metadata']['title'],
            extracted_data['metadata']['doc_type']
        )
        
        self.stats['chunks_created'] += len(chunks)
        
        # Save chunks with complete metadata
        chunks_file = self.config.chunks_dir / f"{file_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                'file_path': str(file_path),
                'file_id': file_id,
                'metadata': extracted_data['metadata'],
                'chunks': chunks,
                'extraction_stats': {
                    'total_chunks': len(chunks),
                    'content_length': len(extracted_data['content']),
                    'processing_date': datetime.now().isoformat()
                }
            }, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Created {len(chunks)} chunks for {file_path.name}")
    
    def _extract_content(self, file_path: Path) -> Dict[str, Any]:
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.pdf_extractor.extract_pdf(file_path)
        
        elif suffix in ['.md', '.markdown']:
            return self.md_extractor.extract_markdown(file_path)
        
        elif suffix in self.code_extractor.code_extensions:
            return self.code_extractor.extract_code(file_path)
        
        elif suffix in self.image_extractor.image_extensions:
            return self.image_extractor.extract_image_text(file_path)
        
        elif suffix in ['.txt', '.rst', '.log']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                return {
                    'metadata': {
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'title': file_path.stem,
                        'doc_type': self._classify_text_type(file_path)
                    },
                    'content': content
                }
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {str(e)}")
                return None
        
        else:
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return None
    
    def _classify_text_type(self, file_path: Path) -> str:
        path_parts = str(file_path).lower()
        
        if any(keyword in path_parts for keyword in ['writeup', 'solution', 'walkthrough']):
            return 'writeup'
        elif any(keyword in path_parts for keyword in ['challenge', 'ctf']):
            return 'challenge'
        elif any(keyword in path_parts for keyword in ['log', 'output']):
            return 'log'
        else:
            return 'text'
    
    def _save_processing_summary(self):
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'config': {
                'max_chunk_size': self.config.max_chunk_size,
                'overlap_size': self.config.overlap_size,
                'enable_ocr': self.config.enable_ocr,
                'enable_code_extraction': self.config.enable_code_extraction
            },
            'next_steps': [
                "Run chunk_processor.py to generate Q&A pairs from chunks",
                "Chunks are saved in processed/chunks/ directory",
                "Each chunk file contains metadata and text chunks ready for Q&A generation"
            ]
        }
        
        summary_file = self.config.metadata_dir / "extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Extraction summary saved to {summary_file}")

def main():
    config = ExtractionConfig()
    
    if not config.data_dir.exists():
        logger.error(f"Data directory not found: {config.data_dir}")
        logger.error("Please create the data directory and add your source files")
        return
    
    extractor = DataExtractor(config)
    
    results = extractor.extract_all_data()
    
    print(f"Files processed: {results['files_processed']}")
    print(f"Files skipped (already processed): {results['skipped_existing']}")
    print(f"Chunks created: {results['chunks_created']}")
    print(f"Errors encountered: {results['errors']}")
    print(f"\nChunks saved in: {config.chunks_dir}")
if __name__ == "__main__":
    main()