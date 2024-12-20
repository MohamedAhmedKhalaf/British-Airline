from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

analyzer = SentimentIntensityAnalyzer()

def preprocess_and_tokenize(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def handle_negations(text):
    negations = ["not", "never", "no", "nothing", "nowhere", "none", "neither", "nor"]
    words = text.split()
    negated = False
    processed_text = []
    
    for word in words:
        if word in negations:
            negated = True
        elif negated and word not in stopwords.words('english'):
            processed_text.append(f"not_{word}")
            negated = False
        else:
            processed_text.append(word)
    
    return ' '.join(processed_text)

def calculate_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    score = sentiment['compound']
    
    if score > 0.05:
        polarity = 'Positive'
    elif score < -0.05:
        polarity = 'Negative'
    else:
        polarity = 'Neutral'
    
    return score, polarity

def predict_sentiment(new_review):
    score, polarity = calculate_sentiment(new_review)
    processed_review = preprocess_and_tokenize(new_review)
    processed_review_with_negations = handle_negations(processed_review)

    return {
        "review": new_review,
        "processed_review": processed_review_with_negations,
        "sentiment_score": score,
        "sentiment_polarity": polarity
    }

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = request.form['review_text']
        result = predict_sentiment(review_text)
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)