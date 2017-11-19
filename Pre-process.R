# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews
library(tm)
library(class)
library(tidyr) #unite two columns into one
#library(ngram) 

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
#setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

# LOAD IN DATA
data = read.csv("Reviews.csv")

#Data Exploration [clean data, cor] -------------------------------------------------------------------------
# Get rid of unuseful information 
names(data)
data_clean = data[, !names(data) %in% c("UserId", "ProfileName") , drop=F]

#create a new column for combined text
data_clean$comb_text<-paste(data_clean$Summary,data_clean$Text, sep=".")

# Create a new column for helpfulness percent
data_clean$help_perc = data_clean$HelpfulnessNumerator/data_clean$HelpfulnessDenominator

# Create a new column for helpfulness indicator
data_clean$help_int<-NaN
data_clean[data_clean$HelpfulnessDenominator>=3 & data_clean$help_perc>0.5,]$help_int<-1
data_clean[data_clean$help_int!=1,]$help_int<-0

# DATA EXPLORATION
hist(data_clean$help_int, breaks="FD", xlim=c(0,1))
hist(data_clean$Score)
cor(data_clean$help_int, data_clean$Score) #cor=0.04234355. not much correlation.

#Pre-Process Text [tfidf, tm_map, sparsity]-------------------------------------------------------------------------

text.df = as.data.frame(data_clean$comb_text, stringsAsFactors = FALSE)
text = VCorpus(DataframeSource(text.df))

# regular indexing returns a sub-corpus
inspect(text[1:10])

# double indexing accesses actual documents
text[[1]]
text[[1]]$content

# compute TF-IDF matrix and inspect sparsity
text.tfidf = DocumentTermMatrix(text, control = list(weighting = weightTfIdf))
text.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
            # sparsity is number of non-zero cells divided by number of zero cells.

#Reducing Term Sparsity 
text.clean = tm_map(text, stripWhitespace)                          # remove extra whitespace
text.clean = tm_map(text.clean, removeNumbers)                      # remove numbers
text.clean = tm_map(text.clean, removePunctuation)                  # remove punctuation
text.clean = tm_map(text.clean, content_transformer(tolower))       # ignore case
text.clean = tm_map(text.clean, removeWords, stopwords("english"))  # remove stop words
text.clean = tm_map(text.clean, stemDocument)                       # stem all words
dtm = DocumentTermMatrix(text.clean) #word count (max term length: 139)
dtm

text.clean.tfidf = DocumentTermMatrix(text.clean, control = list(weighting = weightTfIdf))
# clean = as.matrix(text.clean.tfidf) # Error: cannot allocate vector of size 82.2 Gb
tfidf.sparse = removeSparseTerms(text.clean.tfidf, 0.9)
cleansparse = as.matrix(tfidf.sparse)

#train dataset for modeling
trainvar = as.data.frame(cleansparse)
trainscore = t(data_clean$Score)
traindata = cbind(trainvar,trainscore)

###################### Models ###################### ------------------------------------------------------------------------------------
#Logistic regression ------------------------------------------------------------------------
