## ''''' @author : Megha Lokanadham '''''

import requests
import urllib2
from bs4 import BeautifulSoup
import re
import pandas as pd


# Intializing lists to store the data scraped 
bookLinks = []
bookTitles = []
bookDescriptions = []
bookAuthors = []
bookGenres = []

# Intialize mainUrl i.e the search page containing all books that need to be scraped
mainUrl = 'https://www.goodreads.com/genres/new_releases/non-fiction'
response = requests.get(mainUrl)
html = response.content

soup = BeautifulSoup(html, "html.parser")

# Locating all div tags containing links to books
divTags = soup.find_all('div', attrs={'class': 'coverRow'})

# Storing the links extracted in a list
for div in divTags:
	links =  div.find_all('a')
	for link in links:
		x = link['href']
		bookLinks.append(x)

mainLink = 'https://www.goodreads.com'

print "Scraping links found......."

# Parsing through each link found to scrape the data needed
for link in bookLinks:

	
	# Appending link extracted with the main link
	bookurl = mainLink + link

	response = requests.get(bookurl)
	bookhtml = response.content

	soup2 = BeautifulSoup(bookhtml, "html.parser")

	# Extracting the book's title and storing it in a list
	title = soup2.find('h1', attrs={'class': 'bookTitle'})
	bookTitle = title.text
	bookTitle = bookTitle.strip('\n')
	bookTitles.append(bookTitle)

	# Extracting the book's description and storing it in a list
	description = soup2.find('span', attrs={'id': re.compile('(freeText)\d+')})
	bookDescriptions.append(description.text)

	# Extracting the book's author and storing it in a list
	author = soup2.find('a', attrs={'class': 'authorName'})
	bookAuthors.append(author.text)

	# Extracting the book's genre and storing it in a list
	genre = soup2.find('a', attrs={'class': 'actionLinkLite bookPageGenreLink'})
	bookGenres.append(genre.text)

# Creating a Pandas data frame to save the extracted data
bookDF = pd.DataFrame({'title': bookTitles,
                       'author': bookAuthors,
                       'description': bookDescriptions,
                       'genre': bookGenres})

# Name the file to save the data frame, make sure to change the file name every time
# you change the mainUrl that you wish to extract beacuse it will rewrite the 
# original file which saved the previous data extracted
fileName = 'books13.csv'

# Saving the data frame to a CSV File
bookDF.to_csv(fileName, encoding='utf-8')

print "Saved data scraped......."

