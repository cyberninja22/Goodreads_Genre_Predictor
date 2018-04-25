
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import operator
import _pickle as pickle


# Intializing Lists to store model names and thier accuracies
modelNames = []
modelAccuracies = []

# Reading the training data made after pre processeing the text
data = pd.read_csv("train_book.csv")

# Splitting the data into predictor variable (summaries) and target variable (genres)
summaries = data['description']
genres = data['genre']

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(summaries, genres, test_size=0.4, random_state=0)


# ====== Building a Naive Bayes Model without parameter tuning =========

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_nb = Pipeline([('vect', CountVectorizer()),
                     	('tfidf', TfidfTransformer()),
                     	('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
                      	])

# Fitting the train dataset to the pipeline defined
text_clf_nb = text_clf_nb.fit(X_train, y_train)

# Predicting on the test datset using the fitted model
predicted_nb = text_clf_nb.predict(X_test)

# Calculating the accuracy
accuracy_nb = np.mean(predicted_nb == y_test)

print("Naive Bayes Accuracy without Grid Search",accuracy_nb)

# Appending the accuracy found to list
modelNames.append("text_clf_nb")
modelAccuracies.append(accuracy_nb)


# ========== Improving the Naive Bayes model with Grid Search by tuning the parameters to find the best one ==============

# Creating the parameter dictonary to test for

parameters_nb = {
				'vect__ngram_range': [(1, 1), (1, 2)],
              	'tfidf__use_idf': (True, False),
             	'clf__alpha': (1e-2, 1e-3)
	
				}

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_nb_grid = Pipeline([('vect', CountVectorizer()),
		                     	('tfidf', TfidfTransformer()),
		                     	('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
		                      	])

# Performing a Grid Search to find the best model with the best parameters
gs_clf_nb = GridSearchCV(text_clf_nb_grid, parameters_nb, n_jobs=-1)

# Fitting the best model found to the train dataset
gs_clf_nb = gs_clf_nb.fit(X_train, y_train)

print("Naive Bayes Best Accuracy with GridSearchCV", gs_clf_nb.best_score_)
print("with parameters for Naive Bayes as ", gs_clf_nb.best_params_)

# Appending the accuracy found to list
modelNames.append("gs_clf_nb")
modelAccuracies.append(gs_clf_nb.best_score_)


# ================== Building a SVM Classifier with Stochastic Gradient Descent Learning =========================

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                    alpha=1e-3, max_iter=5, random_state=42)),
                        ])

# Fitting the train dataset to the pipeline defined
text_clf_svm = text_clf_svm.fit(X_train, y_train)

# Predicting on the test datset using the fitted model
predicted_svm = text_clf_svm.predict(X_test)

# Calculating the accuracy
accuracy_svm = np.mean(predicted_svm == y_test)

print("SVM Accuracy with SGDClassifier ",accuracy_svm)

# Appending the accuracy found to list
modelNames.append("text_clf_svm")
modelAccuracies.append(accuracy_svm)


# ======================= Building a Logistic Regression Model =============================

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_log = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-log', LogisticRegression(multi_class='multinomial', solver='newton-cg')),
                        ])

# Fitting the train dataset to the pipeline defined
text_clf_log = text_clf_log.fit(X_train, y_train)

# Predicting on the test datset using the fitted model
predicted_log = text_clf_log.predict(X_test)

# Calculating the accuracy
accuracy_log = np.mean(predicted_log == y_test)

print("Logistic Regression Accuracy without Cross validation", accuracy_log)

# Appending the accuracy found to list
modelNames.append("text_clf_log")
modelAccuracies.append(accuracy_log)


# ================== Building Logistic Regression with Cross Validation =============================

# Specifing a 7 fold Cross Validation
kf = KFold(n_splits=7)
fold = 1

# Performing the 7 fold Cross Validation
for train_index, test_index in kf.split(summaries, genres):

	# Spliting the dataset based on the test and train index specified y the fold
	X_train, X_test = summaries[train_index], summaries[test_index]
	y_train, y_test = genres[train_index], genres[test_index]

	# Building a pipeline to perform the necessary steps needed to build the model
	text_clf_log_cv = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-log', LogisticRegression(multi_class='multinomial', solver='newton-cg')),
                        ])

	# Fitting the train dataset to the pipeline defined
	text_clf_log_cv = text_clf_log_cv.fit(X_train, y_train)

	# Predicting on the test datset using the fitted model
	predicted_log_cv = text_clf_log_cv.predict(X_test)

	# Calculating the accuracy
	accuracy_log_cv = np.mean(predicted_log_cv == y_test)

	print("Logistic Regression Accuracy with Cross Validation ", fold , accuracy_log_cv)
	fold = fold + 1


# Appending the accuracy found at the end of the cross validation to list
modelNames.append("text_clf_log_cv")
modelAccuracies.append(accuracy_log_cv)


# ==================== Building a Logistic Regression model with Stochastic Gradient Descent Learning ===============================

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_log_sgd = Pipeline([('vect', CountVectorizer()),
                        	('tfidf', TfidfTransformer()),
                        	('clf-svm', SGDClassifier(loss='log', penalty='l2',
                                    alpha=1e-3, max_iter=5, random_state=42)),
                        	])

# Fitting the train dataset to the pipeline defined
text_clf_log_sgd = text_clf_log_sgd.fit(X_train, y_train)

# Predicting on the test datset using the fitted model
predicted_log_sgd = text_clf_log_sgd.predict(X_test)

# Calculating the accuracy
accuracy_log_sgd = np.mean(predicted_log_sgd == y_test)
print("Logistic Regression Accuracy with SGDClassifier ",accuracy_log_sgd)

# Appending the accuracy found to list
modelNames.append("text_clf_log_cv")
modelAccuracies.append(accuracy_log_sgd)



# ================================ Building a Nearest Neighbour Model =================================

# Building a pipeline to perform the necessary steps needed to build the model
text_clf_nn = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-log', NearestCentroid()),
                        ])

# Fitting the train dataset to the pipeline defined
text_clf_nn = text_clf_nn.fit(X_train, y_train)

# Predicting on the test datset using the fitted model
predicted_nn = text_clf_nn.predict(X_test)

# Calculating the accuracy
accuracy_nn = np.mean(predicted_nn == y_test)
print("Neareast Neighbour Accuracy", accuracy_nn)

# Appending the accuracy found to list
modelNames.append("text_clf_nn")
modelAccuracies.append(accuracy_nn)

# ================================ Building a Nearest Neighbour Model with Cross Validation =======================

# Specifing a 7 fold Cross Validation
kf = KFold(n_splits=7)
fold = 1

# Performing the 7 fold Cross Validation
for train_index, test_index in kf.split(summaries, genres):

	# Spliting the dataset based on the test and train index specified y the fold
	X_train, X_test = summaries[train_index], summaries[test_index]
	y_train, y_test = genres[train_index], genres[test_index]

	# Building a pipeline to perform the necessary steps needed to build the model
	text_clf_nn_cv = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-log', NearestCentroid()),
                        ])

	# Fitting the train dataset to the pipeline defined
	text_clf_nn_cv = text_clf_nn_cv.fit(X_train, y_train)

	# Predicting on the test datset using the fitted model
	predicted_nn_cv = text_clf_nn_cv.predict(X_test)

	# Calculating the accuracy
	accuracy_nn_cv = np.mean(predicted_nn_cv == y_test)
	print("Neareast Neighbour Accuracy with Cross Validation", fold ,  accuracy_nn_cv)
	fold = fold + 1

# Appending the accuracy found at the end of the cross validation to list
modelNames.append("text_clf_nn_cv")
modelAccuracies.append(accuracy_nn_cv)


# Finding the model with the highest Accuracy 
index, value = max(enumerate(modelAccuracies), key=operator.itemgetter(1))

print("Model with the highest Accuracy", modelNames[index])
print("and its Accuracy", value)


# Creating a pickle object of the best Text Classifier Model found
pickle.dump(text_clf_svm, open("text_clf_svm.pkl", "wb"))

print("Pickle created for the best Text Classifier Model trained ......")























