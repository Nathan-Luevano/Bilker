import json
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunk_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChunkProcessor:
    def __init__(self):
        self.chunks_dir = Path("processed/chunks")
        self.formatted_dir = Path("processed/formatted")
        self.ollama_url = "http://localhost:11434"
        self.model = "deepseek-r1:32b"
        
        # Stats
        self.stats = {
            'chunks_found': 0,
            'already_processed': 0,
            'successfully_processed': 0,
            'failed': 0,
            'total_qa_pairs': 0
        }

    def find_unprocessed_chunks(self) -> List[Path]:
        unprocessed = []
        
        for chunk_file in self.chunks_dir.glob("*_chunks.json"):
            file_id = chunk_file.stem.replace('_chunks', '')
            formatted_file = self.formatted_dir / f"{file_id}_formatted.json"
            
            # Check if formatted file exists and has Q&A pairs
            needs_processing = True
            if formatted_file.exists():
                try:
                    with open(formatted_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        qa_pairs = data.get('qa_pairs', [])
                        if len(qa_pairs) > 0:
                            needs_processing = False
                            self.stats['already_processed'] += 1
                except:
                    pass  # File corrupted, needs reprocessing
            
            if needs_processing:
                unprocessed.append(chunk_file)
            
            self.stats['chunks_found'] += 1
        
        return unprocessed

    def create_simple_prompt(self, text: str, doc_type: str, title: str) -> str:
        return f"""Convert this cybersecurity content into question-answer pairs for training an AI assistant.

Create Q&A pairs that focus on:
- Technical procedures and methods
- Tools and commands
- Vulnerability concepts
- Step-by-step processes

Format EXACTLY like this:
Q: [specific question]
A: [detailed answer]

Q: [another question]  
A: [another answer]

Document Type: {doc_type}
Title: {title}

Content:
{text}

Q&A Pairs:"""

    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=120  
                )
                
                if response.status_code == 200:
                    result = response.json().get('response', '').strip()
                    if result:
                        return result
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}")
                else:
                    logger.warning(f"API error {response.status_code} on attempt {attempt + 1}")
                
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {str(e)}")
                
            if attempt < max_retries - 1:
                time.sleep(2)
        
        logger.error("All LLM call attempts failed")
        return ""

    def parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        qa_pairs = []
        
        # Split by Q: markers
        parts = response.split('Q:')
        
        for part in parts[1:]:  
            part = part.strip()
            if not part:
                continue
                
            if 'A:' not in part:
                continue
                
            q_part, a_part = part.split('A:', 1)
            question = q_part.strip()
            answer = a_part.strip()
            
            if 'Q:' in answer:
                answer = answer.split('Q:')[0].strip()
            
            if len(question) > 10 and len(answer) > 20:
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })
        
        return qa_pairs

    def process_chunk_file(self, chunk_file: Path) -> bool:
        logger.info(f"Processing: {chunk_file.name}")
        
        try:
            # Load chunk data
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = chunk_data.get('chunks', [])
            metadata = chunk_data.get('metadata', {})
            
            all_qa_pairs = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                text = chunk.get('text', '').strip()
                if not text or len(text) < 50:
                    continue
                
                doc_type = chunk.get('doc_type', 'document')
                title = chunk.get('title', 'Unknown')
                
                prompt = self.create_simple_prompt(text, doc_type, title)
                
                response = self.call_llm(prompt)
                if not response:
                    logger.warning(f"No response for chunk {i} in {chunk_file.name}")
                    continue
                
                # Parse response
                qa_pairs = self.parse_qa_response(response)
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    logger.debug(f"Generated {len(qa_pairs)} Q&A pairs from chunk {i}")
                else:
                    logger.warning(f"Failed to parse Q&A from chunk {i} in {chunk_file.name}")
            
            # Save results
            if all_qa_pairs:
                file_id = chunk_file.stem.replace('_chunks', '')
                formatted_file = self.formatted_dir / f"{file_id}_formatted.json"
                
                output_data = {
                    'source_file': chunk_data.get('file_path', 'unknown'),
                    'metadata': metadata,
                    'qa_pairs': all_qa_pairs,
                    'stats': {
                        'chunks_processed': len(chunks),
                        'qa_pairs_generated': len(all_qa_pairs)
                    }
                }
                
                with open(formatted_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                self.stats['successfully_processed'] += 1
                self.stats['total_qa_pairs'] += len(all_qa_pairs)
                
                logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs for {chunk_file.name}")
                return True
            else:
                logger.warning(f"No Q&A pairs generated for {chunk_file.name}")
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error processing {chunk_file.name}: {str(e)}")
            self.stats['failed'] += 1
            return False

    def process_all_chunks(self):
        # Check Ollama connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Cannot connect to Ollama. Make sure it's running.")
                return
        except:
            logger.error("Cannot connect to Ollama. Make sure it's running on localhost:11434")
            return
        
        # Find unprocessed chunks
        unprocessed_chunks = self.find_unprocessed_chunks()
        
        logger.info(f"Found {len(unprocessed_chunks)} chunks to process")
        logger.info(f"Already processed: {self.stats['already_processed']}")
        
        if not unprocessed_chunks:
            logger.info("All chunks already processed!")
            return
        
        # Process chunks
        for i, chunk_file in enumerate(unprocessed_chunks, 1):
            logger.info(f"Progress: {i}/{len(unprocessed_chunks)}")
            self.process_chunk_file(chunk_file)
            
            # logs updates 
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(unprocessed_chunks)} chunks...")
                logger.info(f"Current stats: {self.stats}")
        
        # Final stats
        logger.info("Processing complete!")
        logger.info(f"Final stats: {self.stats}")

def main():
    processor = ChunkProcessor()
    processor.process_all_chunks()
    
    print("CHUNK PROCESSING COMPLETE")
    print(f"Chunks found: {processor.stats['chunks_found']}")
    print(f"Already processed: {processor.stats['already_processed']}")
    print(f"Successfully processed: {processor.stats['successfully_processed']}")
    print(f"Failed: {processor.stats['failed']}")
    print(f"Total Q&A pairs generated: {processor.stats['total_qa_pairs']}")

if __name__ == "__main__":
    main()