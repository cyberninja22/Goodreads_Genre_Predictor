install.packages("readr")
library(readr)
library(tidyr)
install.packages("tidyverse")
library(tidyverse)
#reading all the files 

book_tags<- read_csv("/Users/sharathg/Desktop/DataSets/goodbooks-10k/book_tags.csv")

books<- read_csv('/Users/sharathg/Desktop/DataSets/goodbooks-10k/books.csv')

ratings<-read_csv('/Users/sharathg/Desktop/DataSets/goodbooks-10k/ratings.csv')

tags_books<-read_csv("/Users/sharathg/Desktop/DataSets/goodbooks-10k/tags.csv")

to_read<- read_csv("/Users/sharathg/Desktop/DataSets/goodbooks-10k/to_read.csv")

#####Checking the str and EDA of the data. 
l<-c(book_tags,books,ratings,tags_books,to_read)
k<-sapply(l, View)
View(books)
View()
for (i in length(l) ){
   View(l[i])
}
#All the above are failed functions . 
View(ratings)
View(tags_books)
View(to_read)
######################

#ratings set 
#this is the average ratings on each book 
ratingsby_id<-ratings%>%group_by(book_id) %>% summarise(avg_rating_on_book = mean(rating))
ratingsby_id
View(ratingsby_id)
#building a new column with avarage rating of each book with the dataframe . 
#Ratings <- ratings %>% group_by(book_id) %>% mutate(ratings ,avg_rating = sum(ratings$rating)/length(rating$rating))
#the above ode did not work due to column size error . 
#counting the number of unique user id's for each book_id 
#unique <-ratings%>%summarise(ratings,n_o_books= n(book_id))
#the bove code did not work ,due to the column error .   

                                                  
# merge or joins in two table ratingsby_id and ratings . 
Ratings <- merge(ratings,ratingsby_id, by="book_id")
head(Ratings)

View(Ratings)
#exporting the data set outside for further analysis in pandas or R  . 

write.table(Ratings ,'/Users/sharathg/Desktop/DataSets/goodbooks-10k/BookRatings',sep = ',')
write.table(ratingsby_id,'/Users/sharathg/Desktop/DataSets/goodbooks-10k/Uniq_Avg_rtg',sep = ',')

################################################
####### Building a unique user id , using ratings dataset 
#this gives the average rating given by each user in goodreads (includes ratings to all books )
uui<-ratings%>%group_by(ratings$user_id)%>%arrange(user_id)%>%summarise( avg_rtng_by_user = mean(rating))
head(uui)
colnames(uui)<-c('user_id','avg_rtng_by_user')
View(uui)
Uni_Usr_rtg<- merge(ratings, uui ,by ="user_id")
View(Uni_Usr_rtg)

#Writing the data set to the directory 
write.table(Uni_Usr_rtg,'/Users/sharathg/Desktop/DataSets/goodbooks-10k/Uni_Usr_rtg',sep = ',')




