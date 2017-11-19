#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC

data<-read.csv("modeldata.csv")
colnames(data)[c(1,2)]<-c("id","help_int")
#target: help_int
#predictors: ~.-Id

set.seed(1)
trainrows<-sample(1:nrow(data),size=1000)

###################### Models ###################### ------------------------------------------------------------------------------------
# Logistic regression (step)------------------------------------------------------------------------
glm1 <- glm(as.factor(help_int)~.-id, data=data[trainrows,], family = binomial(link = 'logit'))
step <- stepAIC(glm1, direction="both")
step$anova

# Logistic regression (caret)------------------------------------------------------------------------
#https://rstudio-pubs-static.s3.amazonaws.com/43302_2d242dbea93b46c98ed60f6ac8c62edf.html
fitControl <- trainControl(method = "cv",
                           number = 5,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

glmBoostModel <- train(Class ~ ., data=trainData, method = "glmboost", metric="ROC", trControl = fitControl, tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))
