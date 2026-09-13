import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df['target'] = df['target'].map({0: 0, 4: 1})  # binary

# Clean text
def clean(text):
    text = re.sub(r'@\w+', '', text)        # remove @mentions
    text = re.sub(r'http\S+', '', text)     # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

df['clean'] = df['text'].apply(clean)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['target'], test_size=0.2, random_state=42)

# Vectorize
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_tr = vec.fit_transform(X_train)
X_te = vec.transform(X_test)

# Train with LinearSVC instead of LogisticRegression
model = LinearSVC(C=1.0, max_iter=2000)
model.fit(X_tr, y_train)

print(classification_report(y_test, model.predict(X_te)))