from transformers import pipeline
import joblib
import os

def save_transformer_pipeline():
    print("🚀 Downloading & Initializing DistilBERT (The Hard Case Solver)...")
    
    # This model is specifically fine-tuned for sentiment
    # It handles sarcasm and negations automatically
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipe = pipeline("sentiment-analysis", model=model_name)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # We save the pipeline object
    joblib.dump(sentiment_pipe, 'models/transformer_pipeline.pkl')
    print("✅ Transformer saved! It is now ready to handle sarcasm.")

if __name__ == "__main__":
    save_transformer_pipeline()