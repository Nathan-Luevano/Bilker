import json
from pathlib import Path
import re
from collections import Counter
from typing import List, Dict, Any

def analyze_dataset_quality():
    formatted_dir = Path("processed/formatted")
    
    if not formatted_dir.exists():
        print("No formatted data found. Run chunk_processor.py first.")
        return
    
    # Metrics tracking
    metrics = {
        'total_pairs': 0,
        'question_lengths': [],
        'answer_lengths': [],
        'doc_types': {},
        'quality_issues': [],
        'question_patterns': Counter(),
        'duplicates': set(),
        'technical_coverage': Counter(),
        'difficulty_levels': Counter()
    }
    
    print("Analyzing Q&A dataset quality...\n")
    
    for file in formatted_dir.glob("*_formatted.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pairs = data.get('qa_pairs', [])
                doc_type = data.get('metadata', {}).get('doc_type', 'unknown')
                source_file = data.get('source_file', 'unknown')
                
                _analyze_file_content(pairs, doc_type, source_file, metrics)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    _generate_quality_report(metrics)

def _analyze_file_content(pairs: List[Dict], doc_type: str, source_file: str, metrics: Dict):
    for pair in pairs:
        question = pair.get('question', '')
        answer = pair.get('answer', '')
        
        # Basic metrics
        metrics['total_pairs'] += 1
        metrics['question_lengths'].append(len(question))
        metrics['answer_lengths'].append(len(answer))
        metrics['doc_types'][doc_type] = metrics['doc_types'].get(doc_type, 0) + 1
        
        # Quality analysis
        _check_qa_quality(question, answer, source_file, metrics)
        
        # Pattern analysis
        _analyze_question_patterns(question, metrics)
        
        # Technical coverage
        _analyze_technical_content(question, answer, metrics)
        
        # Difficulty assessment
        _assess_difficulty(question, answer, metrics)

def _check_qa_quality(question: str, answer: str, source_file: str, metrics: Dict):
    issues = []
    
    # Length issues
    if len(question) < 15:
        issues.append("Question too short")
    if len(answer) < 30:
        issues.append("Answer too short")
    
    # Vague questions
    vague_patterns = ['what is', 'tell me about', 'explain this', 'describe the']
    if any(pattern in question.lower() for pattern in vague_patterns):
        if len(question) < 50:  # Short vague questions are problematic
            issues.append("Vague question")
    
    # Generic answers
    generic_patterns = ['it depends', 'there are many', 'it varies', 'this is important']
    if any(pattern in answer.lower() for pattern in generic_patterns):
        issues.append("Generic answer")
    
    # Duplicate detection
    qa_signature = f"{question[:50]}|{answer[:50]}"
    if qa_signature in metrics['duplicates']:
        issues.append("Potential duplicate")
    else:
        metrics['duplicates'].add(qa_signature)
    
    # Missing technical details
    if 'command' in question.lower() or 'tool' in question.lower():
        if not re.search(r'[`\$#]|\-\-?\w+', answer):  # No code/commands in answer
            issues.append("Missing technical details")
    
    if issues:
        metrics['quality_issues'].append({
            'file': source_file,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'issues': issues
        })

def _analyze_question_patterns(question: str, metrics: Dict):
    question_lower = question.lower()
    
    # Question type patterns
    patterns = {
        'how_to': ['how do', 'how can', 'how to'],
        'what_is': ['what is', 'what are', 'what does'],
        'when_to': ['when do', 'when should', 'when is'],
        'why': ['why do', 'why is', 'why should'],
        'which': ['which tool', 'which method', 'which approach'],
        'where': ['where do', 'where can', 'where is'],
        'list': ['list the', 'name the', 'identify the'],
        'compare': ['difference between', 'compare', 'versus'],
        'troubleshoot': ['error', 'problem', 'issue', 'debug', 'fix']
    }
    
    for pattern_type, phrases in patterns.items():
        if any(phrase in question_lower for phrase in phrases):
            metrics['question_patterns'][pattern_type] += 1
            break
    else:
        metrics['question_patterns']['other'] += 1

def _analyze_technical_content(question: str, answer: str, metrics: Dict):
    combined_text = (question + ' ' + answer).lower()
    
    # Technical categories
    categories = {
        'tools': ['nmap', 'wireshark', 'metasploit', 'burp', 'sqlmap', 'john', 'hashcat', 'aircrack'],
        'protocols': ['http', 'https', 'tcp', 'udp', 'dns', 'ssh', 'ftp', 'smtp'],
        'vulnerabilities': ['sql injection', 'xss', 'csrf', 'rce', 'lfi', 'rfi', 'buffer overflow'],
        'techniques': ['reconnaissance', 'enumeration', 'exploitation', 'privilege escalation', 'lateral movement'],
        'forensics': ['steganography', 'metadata', 'hexdump', 'volatility', 'analysis', 'artifacts'],
        'cryptography': ['encryption', 'hash', 'cipher', 'decrypt', 'key', 'algorithm', 'rsa', 'aes'],
        'web_security': ['cookie', 'session', 'jwt', 'oauth', 'cors', 'csp', 'authentication'],
        'system_security': ['privilege', 'permissions', 'sudo', 'suid', 'kernel', 'rootkit']
    }
    
    for category, terms in categories.items():
        if any(term in combined_text for term in terms):
            metrics['technical_coverage'][category] += 1

def _assess_difficulty(question: str, answer: str, metrics: Dict):
    # Simple heuristics for difficulty
    answer_words = len(answer.split())
    technical_terms = len(re.findall(r'\b(?:exploit|vulnerability|payload|shellcode|reverse|forensic)\b', 
                                    answer.lower()))
    
    if answer_words < 50 and technical_terms < 2:
        metrics['difficulty_levels']['beginner'] += 1
    elif answer_words < 150 and technical_terms < 5:
        metrics['difficulty_levels']['intermediate'] += 1
    else:
        metrics['difficulty_levels']['advanced'] += 1

def _generate_quality_report(metrics: Dict):
    total = metrics['total_pairs']
    if total == 0:
        print("No Q&A pairs found in dataset.")
        return
    
    print(f"DATASET QUALITY REPORT")
    print(f"{'=' * 50}")
    
    # Basic statistics
    print(f"\nBASIC STATISTICS")
    print(f"Total Q&A pairs: {total:,}")
    print(f"Average question length: {sum(metrics['question_lengths'])/len(metrics['question_lengths']):.1f} chars")
    print(f"Average answer length: {sum(metrics['answer_lengths'])/len(metrics['answer_lengths']):.1f} chars")
    
    # Content distribution
    print(f"\nCONTENT DISTRIBUTION")
    for doc_type, count in sorted(metrics['doc_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count/total) * 100
        print(f"  {doc_type}: {count:,} pairs ({percentage:.1f}%)")
    
    # Question patterns
    print(f"\nQUESTION PATTERNS")
    for pattern, count in metrics['question_patterns'].most_common():
        percentage = (count/total) * 100
        print(f"  {pattern.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    # Technical coverage
    print(f"\nTECHNICAL COVERAGE")
    for category, count in metrics['technical_coverage'].most_common():
        percentage = (count/total) * 100
        print(f"  {category.replace('_', ' ').title()}: {count:,} pairs ({percentage:.1f}%)")
    
    # Difficulty distribution
    print(f"\nDIFFICULTY DISTRIBUTION")
    for level, count in metrics['difficulty_levels'].items():
        percentage = (count/total) * 100
        print(f"  {level.title()}: {count:,} pairs ({percentage:.1f}%)")
    
    # Quality issues
    quality_issues = len(metrics['quality_issues'])
    print(f"\nQUALITY ISSUES")
    print(f"Issues found in {quality_issues:,} pairs ({(quality_issues/total)*100:.1f}%)")
    
    if quality_issues > 0:
        issue_types = Counter()
        for issue_data in metrics['quality_issues']:
            for issue in issue_data['issues']:
                issue_types[issue] += 1
        
        print("\nTop quality issues:")
        for issue_type, count in issue_types.most_common(5):
            print(f"  {issue_type}: {count:,} occurrences")
    
    # Quality score
    quality_score = max(0, 100 - (quality_issues/total) * 100)
    print(f"\nOVERALL QUALITY SCORE: {quality_score:.1f}/100")
    
    if quality_score < 70:
        print("\nRECOMMENDATIONS:")
        print("  - Review and improve prompts for better Q&A generation")
        print("  - Add more specific technical details to answers")
        print("  - Ensure questions are specific and actionable")
        print("  - Remove or improve low-quality pairs")

# Legacy function for backward compatibility
def analyze_dataset():
    analyze_dataset_quality()

if __name__ == "__main__":
    analyze_dataset_quality()