# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews
library(tm)
library(class)
library(tidyr) #unite two columns into one
library(dplyr) #sample_n for sampling rows
library(pROC)  #AUC 
#library(ngram) 

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
#setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

# LOAD IN DATA
data = read.csv("Reviews.csv")

# Data Exploration [clean data, cor] -------------------------------------------------------------------------
# Get rid of unuseful information 
names(data)
data_clean = data[, !names(data) %in% c("UserId", "ProfileName") , drop=F]

# create a new column for combined text
data_clean$comb_text<-paste(data_clean$Summary,data_clean$Text, sep=".")

# Create a new column for helpfulness percent
data_clean$help_perc = data_clean$HelpfulnessNumerator/data_clean$HelpfulnessDenominator
data_clean[is.na(data_clean$help_perc)==TRUE,]$help_perc<-0

# Create a new column for helpfulness indicator
data_clean$help_int<-NaN
data_clean[data_clean$HelpfulnessDenominator>=3 & data_clean$help_perc>0.5,]$help_int<-1
data_clean[is.na(data_clean$help_int)==TRUE,]$help_int<-0

# DATA EXPLORATION
hist(data_clean$help_int,  xlim=c(0,1))
hist(data_clean$help_perc, xlim=c(0,1))
hist(data_clean$Score)
cor(data_clean$help_int, data_clean$Score) #cor= -0.07077292. not much correlation.
cor(data_clean$help_perc, data_clean$Score) #cor= 0.02812539. not much correlation.

# Pre-Process Text [tfidf, tm_map, sparsity]-------------------------------------------------------------------------
text.df = as.data.frame(data_clean$comb_text, stringsAsFactors = FALSE)
text = VCorpus(DataframeSource(text.df))

#[Check] regular indexing returns a sub-corpus
inspect(text[1:10])

#[Check] double indexing accesses actual documents
text[[1]]
text[[1]]$content

# Remove unuseful charcaters
text.clean = tm_map(text, stripWhitespace)                          # remove extra whitespace
text.clean = tm_map(text.clean, removeNumbers)                      # remove numbers
text.clean = tm_map(text.clean, removePunctuation)                  # remove punctuation
text.clean = tm_map(text.clean, content_transformer(tolower))       # ignore case
text.clean = tm_map(text.clean, removeWords, stopwords("english"))  # remove stop words
text.clean = tm_map(text.clean, stemDocument)                       # stem all words

# compute TF-IDF matrix and inspect sparsity
text.clean.tfidf = DocumentTermMatrix(text.clean, control = list(weighting = weightTfIdf))
#[Check]
text.clean.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
                  # sparsity is number of non-zero cells divided by number of zero cells.
                  # terms: 291780

# clean = as.matrix(text.clean.tfidf) # Error: cannot allocate vector of size 82.2 Gb
tfidf.sparse = removeSparseTerms(text.clean.tfidf, 0.99) #terms: 656

# n-gram -------------------------------------------------------------------------
###  bigrams 
#http://tm.r-forge.r-project.org/faq.html#Bigrams
BigramTokenizer <-
  function(x)
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

tdm <- TermDocumentMatrix(text.clean, control = list(tokenize = BigramTokenizer))
bi.tdm<-removeSparseTerms(tdm, 0.99) #terms: 44
#inspect(bi.tdm)

### trigrams 
BigramTokenizer3 <-
  function(x)
    unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)

tdm3 <- TermDocumentMatrix(text.clean, control = list(tokenize = BigramTokenizer3))
tri.tdm<-removeSparseTerms(tdm3, 0.99) #terms: 0
#inspect(tri.tdm)


# Data for modeling -------------------------------------------------------------------------
combtext.clean.df.uni<-as.data.frame(as.matrix(tfidf.sparse), stringsAsFactors=FALSE)
combtext.clean.df.bi<-as.data.frame(t(as.matrix(bi.tdm)), stringsAsFactors=FALSE)

modeldata<-cbind(data_clean$Id,data_clean$help_int,combtext.clean.df.uni,combtext.clean.df.bi)
#target: help_int
#predictors: ~.-Id

set.seed(1)
trainrows<-sample(1:nrow(modeldata),size=nrow(modeldata)*0.7)

