# Once you have all the csv files that save the data scraped
# merge them using the code below


import csv

fout=open("booksData.csv","a")

# Save the first file
for line in open("books1.csv"):
    fout.write(line)

# Save from the second file to the last, note mention the range
# in the for loop from 2 to (1 + the last number of the file that you wish to save)

# eg: the last file is say books10.csv, then the range is from 2 to 11

for num in range(2,13):
    f = open("books"+str(num)+".csv")
    f.next() # Done to skip the header
    for line in f:
         fout.write(line)
    f.close() 
fout.close()