library(recommenderlab)
library(reshape2)
library(ggplot2)

# Reading in the training dataset
train <- read.csv("ratings.csv",header=TRUE)
books <- read.csv("books.csv",header=TRUE)
books <- books[,c(2,11)]
train <- train[,1:3]

# Generating a train dataset for only the first 1000 datapoints
train2 <- train[1:1000,]

# Making the user - item matrix
ratingmat <- dcast(train2, user_id~book_id, value.var = "rating")
ratingmat <- as.matrix(ratingmat[,-1])
ratingmat
rating <- as(ratingmat, "realRatingMatrix")
rating

as(rating, "list")
as(rating, "matrix")


rating_normalized <- normalize(rating)

# Visualizing the ratings and normalized ratings

image(rating, main = "Raw Ratings")       
image(rating_normalized, main = "Normalized Ratings")

# Building recommadation model

recommender_model <- Recommender(rating_normalized, method = "UBCF", param=list(method="Cosine",nn=30))

# Obtain top 10 recommendations for 1st user in dataset
recom <- predict(recommender_model, rating[1], n=10) 
recom_list <- as(recom, "list")

# Printing th top 10 list found for user
recom_result <- matrix(0,10)

for (i in c(1:10)){
   recom_result[i] <- books[recom_list[[1]][i],2]
   print(books[recom_result[i],2])
   test[[i]] <- books[recom_result[i],2]
}



