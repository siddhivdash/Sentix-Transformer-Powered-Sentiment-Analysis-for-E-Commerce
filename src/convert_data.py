import bz2
import pandas as pd
import os

def convert_fasttext_to_csv(input_bz2, output_csv):
    # Get the absolute path to avoid confusion
    full_path = os.path.abspath(input_bz2)
    print(f"🔍 Looking for file at: {full_path}")

    if not os.path.exists(full_path):
        print(f"❌ ERROR: File not found at {full_path}")
        print("💡 TIP: Make sure the file is inside the 'data' folder and named correctly.")
        return

    labels = []
    texts = []
    
    print(f"📦 Decompressing and parsing {input_bz2}...")
    
    with bz2.open(full_path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split(' ', 1)
            if len(parts) < 2: continue
                
            label = parts[0].strip()
            text = parts[1].strip()
            
            # __label__1 = Negative, __label__2 = Positive
            sentiment = 'negative' if label == '__label__1' else 'positive'
            
            labels.append(sentiment)
            texts.append(text)
            
    df = pd.DataFrame({'text': texts, 'label': labels})
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Saved to {output_csv}")

if __name__ == "__main__":
    # Create data directory if it's missing
    if not os.path.exists('data'):
        os.makedirs('data')

    # Run conversion
    convert_fasttext_to_csv('data/train.ft.txt.bz2', 'data/train.csv')
    convert_fasttext_to_csv('data/test.ft.txt.bz2', 'data/test.csv')