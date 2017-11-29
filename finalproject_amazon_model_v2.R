# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

setwd('/home/yingjie/Desktop/sys6018-final-project')
#setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC
library(pROC) #ROC, AUC

data<-read.csv("modeldata_2.csv")

###################### Data Preprocess ###################### --------------------------------
data<-data[,2:703]
colnames(data)[c(1,2,3,4)]<-c("id","help_int","summary_length","text_length")
data$help_int<-factor(data$help_int)

nrow(data[data$help_int==1,]) #70887
nrow(data[data$help_int==0,]) #324083
all1<-which(data$help_int==1,arr.ind=TRUE) 
all0<-which(data$help_int==0,arr.ind=TRUE) 

# create proper level names to prevent error for modeling
feature.names=names(data)
for (f in feature.names) {
  if (class(data[[f]])=="factor") {
    levels <- unique(c(data[[f]]))
    data[[f]] <- factor(data[[f]],
                        labels=make.names(levels))}
}

# split train and test set.
set.seed(1)
trainrows<-sample(1:nrow(data),size=10000)
trainrows.bal<-c(sample(all1,size=5000),sample(all0,size=5000)) #have not use yet.
trainSet <- data[trainrows,]
testSet <- data[-trainrows,]

# save outcome's name and predictors'names
outcomeName<-'help_int'
predictorsNames<-names(trainSet)[!names(trainSet) %in% c(outcomeName,'id')]

###################### Models ###################### -----------------------------------------
# cross validation setting:
objControl <- trainControl(method='cv', number=2, 
                           returnResamp='none', summaryFunction = twoClassSummary, 
                           classProbs = TRUE,verboseIter=FALSE,
                           allowParallel= TRUE)
# knn (caret) ------------------------------------------------------------------------
# not able to run 10000
model_knn <- train(trainSet[c(1:5000),predictorsNames], trainSet[c(1:5000),outcomeName], 
                   method='knn', 
                   trControl=objControl, 
                   metric = "ROC",
                   tuneLength = 20)
# k-Nearest Neighbors 

# 5000 samples
# 700 predictors
# 2 classes: 'X1', 'X2' 

# No pre-processing
# Resampling: Cross-Validated (2 fold) 
# Summary of sample sizes: 2500, 2500 
# Resampling results across tuning parameters:
  
#   k   ROC        Sens       Spec       
# 5  0.5427568  0.9484612  0.090507726
# 7  0.5465180  0.9592086  0.051876380
# 9  0.5511107  0.9736199  0.040838852
# 11  0.5581212  0.9833903  0.034216336
# 13  0.5607404  0.9877870  0.028697572
# 15  0.5659421  0.9899853  0.027593819
# 17  0.5747573  0.9909624  0.019867550
# 19  0.5758004  0.9907181  0.019867550
# 21  0.5856713  0.9916952  0.017660044
# 23  0.5917719  0.9934050  0.011037528
# 25  0.5930784  0.9938935  0.012141280
# 27  0.5948367  0.9943820  0.011037528
# 29  0.5979787  0.9948705  0.009933775
# 31  0.5957555  0.9956033  0.008830022
# 33  0.6003889  0.9946263  0.007726269
# 35  0.5996764  0.9946263  0.006622517
# 37  0.5999611  0.9948705  0.008830022
# 39  0.6005310  0.9948705  0.004415011
# 41  0.6043820  0.9948705  0.003311258
# 43  0.6044192  0.9965804  0.003311258

# ROC was used to select the optimal model using  the largest value.
# The final value used for the model was k = 43.
# plot(model_knn)
