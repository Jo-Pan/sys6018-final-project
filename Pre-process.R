# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

library(tm)
library(class)
library(ngram)

setwd('/home/yingjie/Desktop/SYS_Final_Proj')

# LOAD IN DATA
data = read.csv("Reviews.csv")

# CLEAN DATA
# Get rid of unuseful information 
names(data)
col = c("ProductId", "UserId", "ProfileName")
data = data[, !names(data) %in% col, drop=F]
# Get rid of 0 helpfulness denominator and only keep helpfulness denominator >= 3
data_clean = data[data$HelpfulnessDenominator!=0 & data$HelpfulnessDenominator>=3,]

# Create a new column for helpfulness percent
data_clean$help_perc = data_clean$HelpfulnessNumerator/data_clean$HelpfulnessDenominator

# DATA EXPLORATION
hist(data_clean$help_perc, breaks="FD", xlim=c(0,1))
hist(data_clean$Score)
ds = data_clean[data_clean$Score == 5,][1:10,]
dp = data_clean[data_clean$Score == 1,][1:10,]
# It seems like high score corresponds to high % usefulness and low score corresponds to low % usefulness
cor(data_clean$help_perc, data_clean$Score) # 0.5033756 correlation

######### PRE-PROCESS TEXT
text.df = as.data.frame(data_clean[,'Text'], stringsAsFactors = FALSE)
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

##### Reducing Term Sparsity #####
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
trainvar = as.data.frame(cleansparse)
traindata = cbind(trainvar,data_clean$Score)
trainscore = t(data_clean$Score)

