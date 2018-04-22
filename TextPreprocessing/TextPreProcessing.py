import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Read data file but ensure the file is in the working directory before running the code
data = pd.read_csv("recommender_data.csv", encoding = 'utf8')

# Extracting the decription and genre labels from the data file
summary = data['description']
genre = data['genre']

#Intializing a list to store the preprocessed text
summaryWords = []

print "Preprocessing Text......."

# Looping through the each decription and preprocessing the text within
for description in summary:
	
	# Tokenizing the words
	tokens = word_tokenize(description)

	# Converting all tokens to lowercase
	tokens = [w.lower() for w in tokens]

	# Remove punctuation from each word
	table = dict.fromkeys(range(32))
	stripped = [w.translate(table) for w in tokens]

	# Remove remaining tokens that are not alphabetic
	words = [word for word in stripped if word.isalpha()]

	# Filtering out stop words
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]

	# Appending preprocessed text to a list
	summaryWords.append(words)

# Storing the preprocessed text to a pandas dataframe
bookDF = pd.DataFrame({'description': summaryWords,
                       'genre': genre})

fileName = 'train_book.csv'

# Storing the data frame to a CSV file
bookDF.to_csv(fileName, encoding='utf-8')

print "Saved Preprocessed Data in train_book.csv file......."
