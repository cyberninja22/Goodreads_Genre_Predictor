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



class HelloForm(Form):
	sayhello = TextAreaField('',[validators.DataRequired()])

@app.route("/")

def index():
	form = HelloForm(request.form)
	return flask.render_template('index.html', form = form)


# Implements the Text classfier model

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

# Implements the Sentiment analysis model

@app.route('/predict_sentiment', methods=['POST'])


def predict_sentiment():

	form = HelloForm(request.form)

	if request.method == 'POST':

		loader = []
		description = request.form['sayhello']

        
		tokens = word_tokenize(description)
		tokens = [w.lower() for w in tokens]
		table = dict.fromkeys(range(32))
		stripped = [w.translate(table) for w in tokens]
		words = [word for word in stripped if word.isalpha()]
		loader.append(words)

		entry = pd.Series(loader)
		x = entry.iloc[0]
		empty = " "
		for i in x:
			empty+= i
			empty += ','
		df = pd.Series(empty)
		print(df.iloc[0:1])

		prediction = modelS.predict(df.iloc[0:1])
		labelS = prediction

	return render_template('result.html', name = labelS)


# Reads the pickle files for the models trained in the main fucntion

if __name__ == '__main__':
	model = pickle.load(open('model/logReg.pkl', "rb"))
	modelS = pickle.load(open('model/best_nb_bigram.pkl', "rb"))
	app.run(host='0.0.0.0', port=8000, debug=True)


