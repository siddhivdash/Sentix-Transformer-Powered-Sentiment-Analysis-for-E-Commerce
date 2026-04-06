import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

def run_training():
    print("🚀 Starting ByteDraft Training...")
    
    # Load the converted CSVs
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        print("❌ Error: CSV files not found. Run src/convert_data.py first!")
        return

    train_df = pd.read_csv('data/train.csv').dropna()
    test_df = pd.read_csv('data/test.csv').dropna()

    # To save time during testing, you can sample the data
    # (Amazon dataset is huge: 3.6 million rows!)
    train_df = train_df.sample(50000, random_state=42)
    test_df = test_df.sample(10000, random_state=42)

    print(f"🧹 Cleaning {len(train_df)} training reviews...")
    train_df['cleaned'] = train_df['text'].apply(clean_text)
    test_df['cleaned'] = test_df['text'].apply(clean_text)

    # Build Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
    ])

    print("🧠 Training on Amazon Polarity Dataset...")
    pipeline.fit(train_df['cleaned'], train_df['label'])

    # Evaluate
    y_pred = pipeline.predict(test_df['cleaned'])
    acc = accuracy_score(test_df['label'], y_pred)
    
    print(f"\n✨ FINAL ACCURACY: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(test_df['label'], y_pred))

    # Save
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(pipeline, 'models/sentiment_pipeline.pkl')
    print("📦 Pipeline saved successfully!")

if __name__ == "__main__":
    run_training()