import json
from pathlib import Path

def analyze_dataset():
    formatted_dir = Path("processed/formatted")
    
    total_pairs = 0
    question_lengths = []
    answer_lengths = []
    doc_types = {}
    
    for file in formatted_dir.glob("*_formatted.json"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pairs = data.get('qa_pairs', [])
            doc_type = data.get('metadata', {}).get('doc_type', 'unknown')
            
            total_pairs += len(pairs)
            doc_types[doc_type] = doc_types.get(doc_type, 0) + len(pairs)
            
            for pair in pairs:
                question_lengths.append(len(pair['question']))
                answer_lengths.append(len(pair['answer']))
    
    print(f"Total Q&A pairs: {total_pairs}")
    print(f"Avg question length: {sum(question_lengths)/len(question_lengths):.1f} chars")
    print(f"Avg answer length: {sum(answer_lengths)/len(answer_lengths):.1f} chars")
    print("\nContent distribution:")
    for doc_type, count in sorted(doc_types.items()):
        print(f"  {doc_type}: {count} pairs ({count/total_pairs*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset()