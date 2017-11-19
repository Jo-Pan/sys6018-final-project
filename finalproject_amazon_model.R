# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC

data<-read.csv("modeldata.csv")
colnames(data)[c(1,2)]<-c("id","help_int")
data$help_int<-factor(data$help_int)

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
trainSet <- data[ trainrows,]
testSet <- data[-trainrows,]

# save outcome's name and predictors'names
outcomeName<-'help_int'
predictorsNames<-names(trainSet)[!names(trainSet) %in% c(outcomeName,'id')]

###################### Models ###################### ------------------------------------------------------------------------------------
# Logistic regression - step (not able to run)------------------------------------------------------------------------
glm1 <- glm(as.factor(help_int)~.-id, data=data[trainrows,], family = binomial(link = 'logit'))
step <- stepAIC(glm1, direction="backward")
step$anova


# Logistic regression - step (caret) (bad) ------------------------------------------------------------------------
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],
                 trControl=objControl,method='glmStepAIC')

#Plotting Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")

# gbm (boosting) (caret)------------------------------------------------------------------------
# cross validation setting:
objControl <- trainControl(method='cv', number=2, 
                           returnResamp='none', summaryFunction = twoClassSummary, 
                           classProbs = TRUE,verboseIter=FALSE,
                           allowParallel= TRUE)

# parameters optimizing:
#     history: 
#       1) with 1000 rows: "shrinkage = c(0.01,0.001)".        0.001 is much better.
#                          "interaction.depth =  c(1, 5, 10)". 1 & 5 is better. 
#                          "n.trees = c(100,500,1000)".       500 the best.slightly outperform 100 sometimes.
#                                                             100 best among the rest.
#                           best auc = 0.542
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5),
                        n.trees = c(100,500,1000),
                        shrinkage = c(0.001),
                        n.minobsinnode=10)

Model_gbm <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  tuneGrid = gbmGrid)

plot(Model_gbm)
