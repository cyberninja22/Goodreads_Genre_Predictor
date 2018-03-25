# Goodreads_Genre_Predictor
 CSP_571_Project 

This project is a text classification project. The idea of the project is to classify the genres of books based on the text summaries of the book provided by the user. The classification is going to be a multi class classification. The project is divided into 4 phases,
	
  - Data Scraping
	- Data Preprocessing
	- Text Classification Modeling
	- Building a User Interface to use the classifier.


Team Members : 
   	-  Sharath Gangalapadu.
     - Megha Lokanadham
	 
Project Proposal: 

   -The objective of the project is to classify books based on genre using text summaries from books belonging to genres such as fiction, drama, romance, adventure etc. 
	- The preliminary data source we plan on using is goodreads.com.
	- The text classification would be done using a text classifier model such as Naive Bayes classifier, this would be the model we would start with.
	- The success metrics we plan on using to evaluate the performance of the model is accuracy and precision.


Project Plan: The plan to proceed about doing the project is as follows,


- Data Scraping: 
		The first step to the project would be to scrape the goodreads website and extract information such as, book summaries, genre, ratings and number of ratings.
		The tools we plan on using to do this are either python’s beautiful soup, nutch crawler or automate the process by writing code on Selenium.

	- Data Preprocessing: 
		The data extracted needs to have text processing performed on it. To do so, we’ll eliminate the stop words and also tokenize the text. And this will be done writing a Python code using the scikit-learn library.


	- Text Classification: 
		Text classification is the main task of the project and we plan on splitting the data obtained from scrapping into test and train, or either use a cross validation technique to train the model. The model we plan on using is a Naive Bayes 			Classifier, this although might change if we don’t get good results from it.

	- Evaluation:
		The success metrics we plan on using to evaluate the models performance is accuracy and precision. If we end up testing and training the data on different models, we’ll use the model that performs the best based on these two metrics.
		

	- User Interface:
		The trained classifier will be used with a simple user interface, wherein the user input a text summary of a book he wishes to find the genre of and provide the 	output as the genre the model predicts.
	

















