from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Sentiment Analysis Setup
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

# Load your machine learning models
knn_pipeline = joblib.load('KNN_model.joblib')
svm_pipeline = joblib.load('SVM_model.joblib')
decision_tree_pipeline = joblib.load('Decision_Tree_model.joblib')
naive_bayes_pipeline = joblib.load('Naive_Bayes_model.joblib')
aircraft_types = [
        "Boeing 777",
        "Boeing 787",
        "Unknown",
        "Airbus A320",
        "Airbus A321",
        "Airbus A350",
        "Airbus A380",
        "Embraer 190",
        "Boeing 747",
        "Boeing 737",
     ]
# Load the label encoders
label_encoders = joblib.load('label_encoders.joblib')
# Define the data loading and preprocessing
def preprocess_input(input_data, label_encoders):
    
    for col in label_encoders.keys():
        if col in input_data:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]
    X = pd.DataFrame(input_data, index=[0])
    return X
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = request.form['review_text']
        result = predict_sentiment(review_text)
        return render_template('index.html', result=result)
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles the prediction page using saved models."""
    if request.method == 'POST':
        # Extract user inputs from the form
        input_data = {
            'aircraft' : request.form['aircraft'],
            'seating_comfort': float(request.form['seating_comfort']),
            'staff_service': float(request.form['staff_service']),
            'food_quality': float(request.form['food_quality']),
            'entertainment': float(request.form['entertainment']),
            'wifi': float(request.form['wifi']),
            'ground_service': float(request.form['ground_service']),
            'value_for_money': float(request.form['value_for_money']),

        }

        # Preprocess inputs
        X = preprocess_input(input_data, label_encoders)


        # Perform prediction using each model
        knn_prediction = knn_pipeline.predict(X)[0]
        svm_prediction = svm_pipeline.predict(X)[0]
        decision_tree_prediction = decision_tree_pipeline.predict(X)[0]
        naive_bayes_prediction = naive_bayes_pipeline.predict(X)[0]

        # Get the predictions based on the model
        knn_prediction_label = "Recommended" if knn_prediction == 1 else "Not Recommended"
        svm_prediction_label = "Recommended" if svm_prediction == 1 else "Not Recommended"
        decision_tree_prediction_label = "Recommended" if decision_tree_prediction == 1 else "Not Recommended"
        naive_bayes_prediction_label = "Recommended" if naive_bayes_prediction == 1 else "Not Recommended"


        # Render the results on the template
        return render_template('predict.html', 
                                knn_prediction=knn_prediction_label,
                                svm_prediction=svm_prediction_label,
                                decision_tree_prediction=decision_tree_prediction_label,
                                naive_bayes_prediction = naive_bayes_prediction_label
                            )
    return render_template('predict.html',aircraft_types=aircraft_types)

if __name__ == '__main__':
    app.run(debug=True)