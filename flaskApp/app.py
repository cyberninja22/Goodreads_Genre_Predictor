## ''''' @author : Megha Lokanadham '''''


import flask
from flask import Flask, render_template, request
import _pickle as pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
from wtforms import Form, TextAreaField, validators
from sklearn.externals import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



app = Flask(__name__)

# def langRec(text):
#     k=detect(text)
#     return k

def thresh(num):
	sentiment=[]
	if num>=0.1:
		sentiment.append('Positive')
	elif -0.1<=num<=0.1:
		sentiment.append('Neutral')
	else:
		sentiment.append('Negative')
	return(sentiment[0])

def sentimentA(text):
	analyser = SentimentIntensityAnalyzer()
	snt = analyser.polarity_scores(text)
	return snt['compound']



class HelloForm(Form):
	sayhello = TextAreaField('',[validators.DataRequired()])

@app.route("/")

def index():
	form = HelloForm(request.form)
	return flask.render_template('index.html', form = form)

@app.route('/predict', methods=['POST'])

def make_prediction():

	form = HelloForm(request.form)

	if request.method == 'POST':

		loader = []
		description = request.form['sayhello']
        
		tokens = word_tokenize(description)
		tokens = [w.lower() for w in tokens]
		table = dict.fromkeys(range(32))
		stripped = [w.translate(table) for w in tokens]
		words = [word for word in stripped if word.isalpha()]
		stop_words = set(stopwords.words('english'))
		words = [w for w in words if not w in stop_words]
		loader.append(words)

		entry = pd.Series(loader)
		x = entry.iloc[0]
		empty = " "
		for i in x:
			empty+= i
			empty += ','
		df = pd.Series(empty)

		prediction = model.predict(df.iloc[0:1])
		label = prediction
		
		return render_template('result.html', name = label)

@app.route('/predict_sentiment', methods=['POST'])


def predict_sentiment():

	form = HelloForm(request.form)

	text = request.form['sayhello']

	sentiment_score = sentimentA(text)

	sentiment = thresh(sentiment_score)

	return render_template('result.html', name = sentiment)


if __name__ == '__main__':
	model = pickle.load(open('model/logReg.pkl', "rb"))
	# model = joblib.load()
	app.run(host='0.0.0.0', port=8000, debug=True)


