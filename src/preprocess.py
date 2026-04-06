import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
# Keep words that define sentiment
neg_words = {'not', 'no', 'never', 'neither', 'nor', 'but', 'however'}
stop_words = set(stopwords.words('english')) - neg_words

def clean_text(text):
    if not isinstance(text, str): return ""
    # Remove special chars but keep exclamation marks for sentiment intensity
    text = re.sub(r'[^a-zA-Z\s!]', '', text).lower()
    tokens = text.split()
    # Lemmatize (e.g., 'running' -> 'run')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)