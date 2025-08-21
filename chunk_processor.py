import json
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import re

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
        
        # Create formatted directory
        self.formatted_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced stats tracking
        self.stats = {
            'chunks_found': 0,
            'already_processed': 0,
            'successfully_processed': 0,
            'failed': 0,
            'total_qa_pairs': 0,
            'quality_filtered': 0,
            'avg_pairs_per_chunk': 0.0
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

    def create_advanced_prompt(self, text: str, doc_type: str, title: str, chunk_id: int = 0) -> str:
        type_instructions = {
            'writeup': "Focus on exploitation techniques, vulnerability analysis, and step-by-step attack methodologies. Include specific commands, tools used, and critical decision points.",
            'code': "Analyze the code for security implications, functionality, and potential vulnerabilities. Create questions about implementation details, security considerations, and usage patterns.",
            'challenge': "Extract challenge-solving strategies, key insights, and transferable knowledge. Focus on methodology, common pitfalls, and solution approaches.",
            'research_paper': "Emphasize theoretical concepts, experimental methodology, and practical applications. Create questions that test understanding of core principles.",
            'documentation': "Focus on practical usage, configuration details, and operational knowledge. Create questions that help users apply the information effectively.",
            'reference': "Extract key facts, procedures, and quick-reference information. Create questions that test recall and application of critical details.",
            'log': "Analyze patterns, indicators, and forensic evidence. Create questions about detection, analysis techniques, and incident response."
        }
        
        specific_instruction = type_instructions.get(doc_type, "Create comprehensive questions that extract key knowledge and practical insights from the content.")
        
        return f"""# CYBERSECURITY KNOWLEDGE EXTRACTION TASK

You are an expert cybersecurity educator creating high-quality training data. Your task is to convert the provided content into precise, educational question-answer pairs.

## DOCUMENT CONTEXT
- **Type**: {doc_type}
- **Title**: {title}
- **Chunk**: {chunk_id + 1}

## SPECIALIZED INSTRUCTIONS FOR {doc_type.upper()}
{specific_instruction}

## QUALITY REQUIREMENTS

### Questions Must:
1. **Be Specific**: Target concrete concepts, not vague generalizations
2. **Test Understanding**: Go beyond surface-level recall to application and analysis
3. **Use Precise Language**: Include technical terms and specific contexts
4. **Be Self-Contained**: Understandable without the source document
5. **Avoid Ambiguity**: Have one clear, correct answer

### Answers Must:
1. **Be Comprehensive**: Cover all relevant aspects of the question
2. **Include Context**: Explain why something works, not just what it does
3. **Use Examples**: Provide concrete examples, commands, or scenarios when applicable
4. **Be Actionable**: Give practical steps or implementation details
5. **Maintain Accuracy**: Be technically precise and factually correct

## QUESTION TYPES TO CREATE

**Prioritize these question patterns:**
- "How do you [perform specific action]?"
- "What is the purpose of [specific tool/technique]?"
- "When would you use [specific approach]?"
- "What are the key steps to [accomplish task]?"
- "How can you identify [specific indicator/pattern]?"
- "What security implications arise from [specific scenario]?"

## OUTPUT FORMAT

Generate 3-8 high-quality Q&A pairs using this EXACT format:

```
Q: [Specific, focused question]
A: [Comprehensive, detailed answer with examples and context]

Q: [Another specific question]
A: [Another comprehensive answer]
```

## SOURCE CONTENT

{text}

## GENERATED Q&A PAIRS
"""

    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  
                        "top_p": 0.85,       
                        "top_k": 40,         
                        "repeat_penalty": 1.1,  
                        "presence_penalty": 0.0,
                        "frequency_penalty": 0.0,
                        "mirostat": 2,        
                        "mirostat_tau": 5.0,
                        "num_predict": 2048   
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=120  
                )
                
                if response.status_code == 200:
                    result = response.json().get('response', '').strip()
                    if result and len(result) > 50:
                        return result
                    else:
                        logger.warning(f"Insufficient response on attempt {attempt + 1}: {len(result)} chars")
                else:
                    logger.warning(f"API error {response.status_code} on attempt {attempt + 1}: {response.text[:200]}")
                
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {str(e)}")
                
            if attempt < max_retries - 1:
                time.sleep(2)
        
        logger.error("All LLM call attempts failed")
        return ""

    def parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        qa_pairs = []
        
        response = response.replace('```', '').strip()
        
        # Split by Q: markers (case insensitive)
        parts = re.split(r'\n?Q:', response, flags=re.IGNORECASE)
        
        for part in parts[1:]:  
            part = part.strip()
            if not part:
                continue
                
            # Look for A: marker (case insensitive)
            a_match = re.search(r'\n?A:', part, flags=re.IGNORECASE)
            if not a_match:
                continue
                
            question = part[:a_match.start()].strip()
            answer = part[a_match.end():].strip()
            
            # Clean up next Q: if it leaked into this answer
            next_q = re.search(r'\n?Q:', answer, flags=re.IGNORECASE)
            if next_q:
                answer = answer[:next_q.start()].strip()
            
            # Quality validation
            if self._validate_qa_pair(question, answer):
                qa_pairs.append({
                    'question': self._clean_text(question),
                    'answer': self._clean_text(answer)
                })
        
        return qa_pairs
    
    def _validate_qa_pair(self, question: str, answer: str) -> bool:
        # Basic length requirements
        if len(question) < 15 or len(answer) < 30:
            return False
            
        # Check for meaningful content (not just placeholder text)
        placeholders = ['[insert', '[describe', '[explain', 'lorem ipsum', 'example text']
        if any(placeholder in question.lower() or placeholder in answer.lower() 
               for placeholder in placeholders):
            return False
            
        # Ensure question ends with question mark or is imperative
        question_indicators = ['?', 'how', 'what', 'when', 'where', 'why', 'which', 
                              'describe', 'explain', 'list', 'identify']
        if not any(indicator in question.lower() for indicator in question_indicators):
            return False
            
        # Check answer has substantive content
        answer_words = answer.split()
        if len(answer_words) < 8:  # Minimum 8 words for substantial answer
            return False
            
        # Avoid very repetitive content
        unique_words = set(answer_words)
        if len(unique_words) / len(answer_words) < 0.3:  # Too repetitive
            return False
            
        return True
    
    def _clean_text(self, text: str) -> str:
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common artifacts
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Ensure proper sentence ending
        if not text.endswith(('.', '!', '?', ':', ';')):
            text += '.'
            
        return text.strip()

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
                
                prompt = self.create_advanced_prompt(text, doc_type, title, i)
                
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