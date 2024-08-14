
from flask import Flask,render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import requests
import re
from collections import Counter
import os
from dotenv import load_dotenv
import warnings


warnings.filterwarnings('ignore')

load_dotenv()

flag = True

if flag == True:
    data = pd.read_csv('cleaned_WELFake_Dataset.csv')

    data.iloc[:70000,:]

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_data['label'])
    test_labels_encoded = label_encoder.transform(test_data['label'])

    train_data['label'] = train_labels_encoded
    test_data['label'] = test_labels_encoded

    vectorizer = TfidfVectorizer()
    requiredText = train_data['cleaned_title'].values + ' ' + train_data['cleaned_text'].values
    requiredTarget = test_data['label'].values
    vectorizer.fit(requiredText)
    wordFeatures = vectorizer.transform(requiredText)


    print('Training Naive Bayes...')
    nb_classifier = make_pipeline(vectorizer, MultinomialNB())
    nb_classifier.fit(train_data['cleaned_title'] + ' ' + train_data['cleaned_text'], train_data['label'])

    print('Training Passive Aggressive...')
    pa_classifier = make_pipeline(vectorizer, PassiveAggressiveClassifier(max_iter=1000, random_state=42))
    pa_classifier.fit(train_data['cleaned_title'] + ' ' + train_data['cleaned_text'], train_data['label'])

    print('Training Logistic Regression...')
    lr_classifier = make_pipeline(vectorizer, LogisticRegression(max_iter=1000, random_state=42))
    lr_classifier.fit(train_data['cleaned_title'] + ' ' + train_data['cleaned_text'], train_data['label'])
    sentence = "Leave a reply Mary O Malley A friend of mine was gifted a vacation on a river barge through France and Germany She was really excited because she could never afford to do this on her own Her itinerary included a flight from Seattle to New York and the next day she and her cousin would fly to Europe together As they were getting ready to go through the international security checkpoint the TSA officer would not let my friend through because her passport had expired At that moment a feeling of panic came over her when she realized she had grabbed the wrong passport As the panic began to intensify she tuned into her body and used her breath to calm herself down and think rationally Having done awakening work for some time she realized that reacting to whatever Life is offering only creates suffering She said Mary I can t believe it If this happened years ago I would have had a temper tantrum and been in tears all day Instead I calmly told my cousin to go on without me and I would have a friend in Seattle overnight my passport so I could take a flight the following day This is the power of a epting whatever Life brings She added I did not react or panic when I realized I would have to pay for a new ticket and stay in New York by myself It all turned out so differently than what I thought and it was all okay Whenever we are faced with a challenge whether it is a cut on our finger a raging boss or an expired passport we all react in our own unique ways One of the core ways we can heal ourselves is by getting to know our own patterns of reaction Some of the standard modes of reaction are the stoic the pleaser the worrier the rager the freezer the rescuer the victim the judger and the self absorbed one We put these reactive parts of ourselves together into our own particular style Our patterns of reaction can bring so much heartache into our lives My primary mode of reaction was to freeze and like all patterns of reaction it would tighten me isolate me and cut me off from the flow of love that is Life It got stronger as I got older and it became much more entrenched when I tried to make it stop Then I learned how to see it and meet it with understanding and mercy Only then was I able to free up this frozen energy and learn how to respond to Life rather than always living in reaction This process of seeing loving and letting go of your patterns of reaction happens in three phases First phase You are caught You either can t see your patterns or if you do you have little willingness to do anything about them You also don t recognize the consequences that come from living out of your repetitive reactions When you decide you don t want to live this way anymore you usually declare war on the pattern trying to wrestle it to the ground This only works for a short period of time because you haven t done the work needed to dissolve the pattern When the pattern comes back again you then get caught in self judgment I did it again and despair I will never get out of this You begin to move into the second phase when you see that the price you pay for taking care of yourself in your old ways isn t worth it and you understand that trying to wrestle it to the ground doesn t work Second phase As you come to realize that living out of your old patterns isn t how you want to live and trying to get rid of them seems to only empower them you begin to become curious about what is going on and feel the possibility of living another way At the beginning of this phase when your patterns of reaction are triggered you will get lost in them most of the time But slowly you become more curious and more merciful with yourself Even when you get completely lost there comes a time when you can let go of judgment and despair and simply notice how you are reacting This may be right in the middle of the pattern a few minutes afterwards or a few days afterwards You become more curious than controlling and more compassionate than judgmental You finally come to the place where you can actually stand with the discomfort of not following your pattern This almost feels like detox and it is good to have the support of other people as you learn how to be with your experience rather than running away from it The more you can be with yourself the more it opens your heart And that is really what you have been longing for all along the ability to live from your own heart Third phase As your ability to see and let go of your old patterns increases you enter the third phase This is where your primary relationship with yourself is one of compassion and curiosity You still may get caught in your patterns but only for short periods of time and rather than bringing up judgment or despair they become an invitation to open back into Life The more you can meet yourself exactly as you are the more you discover how to live from your own truth You can now let go of the idea that a good life is one where everything is under control Instead you learn how to ride the ups and downs of your life trusting yourself trusting your life You then become the awakened heart that heals not only yourself but also the world There isn t a clear delineation with these three phases On any particular day you may touch into all three phases But the more you can be curious and merciful with yourself the more you will naturally gravitate toward the third phase the place of truly becoming yourself This is the greatest gift you can give to yourself and to Life SF Source Mary O Malley "
    flag = False

def extract_text_from_link(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text.strip() for p in soup.find_all('p') if p.text.strip()])
        return text
    except requests.RequestException as e:
        print("Error fetching content:", e)
    except Exception as e:
        print("Error:", e)
    return None

def is_url(text):
    url_pattern = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    )
    return bool(re.match(url_pattern, text))


# Example usage
# url = "https://www.livemint.com/news/india/poonam-pandey-husband-face-rs-100-crore-defamation-case-for-fake-death-stunt-11707885787401.html"
# article_text = extract_text_from_link(url)
# if article_text:
#     print("Extracted Text:", article_text)
# else:
#     print("Failed to extract text from the provided link.")

app = Flask(__name__)

@app.route('/')
def LandingPage():
    return render_template('home.html')

@app.route('/home')
def hello_world():
    return render_template('home.html')

res = {'prompt' : '', 'data' : 0}
@app.route('/factchecker', methods = ["GET"])
def factchecker():
    res['prompt'] = ''
    res['data'] = 0
    return render_template('factchecker.html', data = res)
    
@app.route('/factchecker', methods = ["POST"])
def result():
    res['prompt'] = request.form['prompt']
    if res['data'] != "" :res['data'] = 1
    else : res['data'] = 0
    print(request.form) 
    if 'prompt' in request.form:
        prompt = request.form['prompt']
        if is_url(prompt):
            prompt = extract_text_from_link(prompt)
            if prompt:
                print("Extracted Text:", prompt)
                nb_predictions = nb_classifier.predict([prompt])
                pa_predictions = pa_classifier.predict([prompt])
                lr_predictions = lr_classifier.predict([prompt])
                ensemble_predictions = []
                for nb_pred, pa_pred, lr_pred in zip(nb_predictions, pa_predictions, lr_predictions):

                    if nb_pred + pa_pred + lr_pred >= 2:
                        ensemble_predictions.append(1)
                    else:
                        ensemble_predictions.append(0)

                # ensemble_accuracy = accuracy_score(test_data['label'], ensemble_predictions)
                # print("Ensemble Accuracy:", ensemble_accuracy)
                
                def ensemble_predict(sentence):
                    nb_prediction = nb_classifier.predict([sentence])[0]
                    pa_prediction = pa_classifier.predict([sentence])[0]
                    lr_prediction = lr_classifier.predict_proba([sentence])[0] * 100

                    predictions = [nb_prediction, pa_prediction, lr_prediction]

                    # Count the votes
                    #vote_counter = Counter(predictions)

                    # Get the most common prediction (majority vote)
                    #majority_vote = vote_counter.most_common(1)[0][0]

                    return lr_prediction

                # ensemble_model = {
                #     'nb_classifier': nb_classifier,
                #     'pa_classifier': pa_classifier,
                #     'lr_classifier': lr_classifier,
                #     # Add any additional components of your ensemble if needed
                # }

                ensemble_prediction = ensemble_predict(prompt)
                print("Ensemble Prediction:", ensemble_prediction)

                summarizer = pipeline("summarization", model="Falconsai/text_summarization")
                summary = summarizer(prompt, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
                print(summary)
                print(type(summary))
                API_URL = "https://api-inference.huggingface.co/models/IMSyPP/hate_speech_en"
                headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}

                def query(payload):
                    response = requests.post(API_URL, headers=headers, json=payload)
                    return response.json()
                    
                output = query(summary)


                label_mapping = {
                    'LABEL_0': 'acceptable',
                    'LABEL_1': 'inappropriate',
                    'LABEL_2': 'offensive',
                    'LABEL_3': 'violent'
                }

                # hate_speech = [{'label': label_mapping[result['label']], 'score': result['score']} for result in output]
                # print(hate_speech)

                # model = pickle.load(open('lr_classifier.pkl', 'rb'))
                # vectors = pickle.load(open('vectorizer.pkl', 'rb'))
                # print(model)
                # print(prompt)
                # print(vectors)
                # input = vectors.transform(['fdsvfdv dsv refdv fdbdgb'])
                # print(input)
                # pred = model.predict(input)[0]
                res['prediction'] = round(ensemble_prediction[1],2)
                res['summary'] = summary
                # res['hate_speech'] = hate_speech
            else:
                print("Failed to extract text from the provided link.")
        else:
            print(type(prompt))
            nb_predictions = nb_classifier.predict([prompt])
            pa_predictions = pa_classifier.predict([prompt])
            lr_predictions = lr_classifier.predict([prompt])
            ensemble_predictions = []
            for nb_pred, pa_pred, lr_pred in zip(nb_predictions, pa_predictions, lr_predictions):

                if nb_pred + pa_pred + lr_pred >= 2:
                    ensemble_predictions.append(1)
                else:
                    ensemble_predictions.append(0)

            # ensemble_accuracy = accuracy_score(test_data['label'], ensemble_predictions)
            # print("Ensemble Accuracy:", ensemble_accuracy)
            
            def ensemble_predict(sentence):
                # Make predictions using each classifier
                nb_prediction = nb_classifier.predict([sentence])[0]
                pa_prediction = pa_classifier.predict([sentence])[0]
                lr_prediction = lr_classifier.predict_proba([sentence])[0] * 100

                # Create a list of predictions
                predictions = [nb_prediction, pa_prediction, lr_prediction]

                # Count the votes
                #vote_counter = Counter(predictions)

                # Get the most common prediction (majority vote)
                #majority_vote = vote_counter.most_common(1)[0][0]

                return lr_prediction

            # ensemble_model = {
            #     'nb_classifier': nb_classifier,
            #     'pa_classifier': pa_classifier,
            #     'lr_classifier': lr_classifier,
            #     # Add any additional components of your ensemble if needed
            # }

            ensemble_prediction = ensemble_predict(prompt)
            print("Ensemble Prediction:", ensemble_prediction)

            summarizer = pipeline("summarization", model="Falconsai/text_summarization")
            summary = summarizer(prompt, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
            print(summary)
            API_URL = "https://api-inference.huggingface.co/models/IMSyPP/hate_speech_en"
            headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
                
            output = query(summary)


            label_mapping = {
                'LABEL_0': 'acceptable',
                'LABEL_1': 'inappropriate',
                'LABEL_2': 'offensive',
                'LABEL_3': 'violent'
            }

            # hate_speech = [{'label': label_mapping[result['label']], 'score': result['score']} for result in output]
            # print(hate_speech)

            # model = pickle.load(open('lr_classifier.pkl', 'rb'))
            # vectors = pickle.load(open('vectorizer.pkl', 'rb'))
            # print(model)
            # print(prompt)
            # print(vectors)
            # input = vectors.transform(['fdsvfdv dsv refdv fdbdgb'])
            # print(input)
            # pred = model.predict(input)[0]
            res['prediction'] = round(ensemble_prediction[1],2)
            res['summary'] = summary
            # res['hate_speech'] = hate_speech
            

        return render_template('factchecker.html', data = res)

    

# @app.route('/factchecker', methods = ["POST"])
# def result():
#     res = request.form['prompt']
#     print(res)
#     return render_template('factchecker.html',data = res )
    

if __name__ == '__main__':
    app.run(debug=True)