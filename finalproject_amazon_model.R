# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC
library(pROC) #ROC, AUC

data<-read.csv("modeldata.csv")

###################### Data Preprocess ###################### --------------------------------
colnames(data)[c(1,2)]<-c("id","help_int")
data$help_int<-factor(data$help_int)

nrow(data[data$help_int=='X2',]) #97890
nrow(data[data$help_int=='X1',]) #470564
all1<-which(data$help_int=='X2',arr.ind=TRUE) 
all0<-which(data$help_int=='X1',arr.ind=TRUE) 

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

# Logistic regression - step (not able to run)------------------------------------------------
glm1 <- glm(as.factor(help_int)~.-id, data=data[trainrows,], family = binomial(link = 'logit'))
step <- stepAIC(glm1, direction="backward")
step$anova


# Logistic regression (caret) -----------------------------------------------------------------
model_glm<-train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                 method='glm')
  #summary(model_glm)
predictions <- predict(object=model_glm, trainSet[,predictorsNames])
auc <- roc(ifelse(trainSet[,outcomeName]=="X2",1,0), ifelse(predictions=="X2",1,0))
print(auc$auc) #0.5367

# Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")
glm_imp<-varImp(object=model_glm)$importance
glm_imp$var<-row.names(glm_imp)
glm_imp<-glm_imp[order(-glm_imp$Overall),]
glm_imp_var<-glm_imp[glm_imp$Overall>10,'var'] #470


# gbm (boosting) (caret)----------------------------------------------------------------------
# history: 
#       1) with 1000 rows: "shrinkage = c(0.01,0.001)".        0.001 is much better.
#                          "interaction.depth =  c(1, 5, 10)". 1 & 5 is better. 
#                          "n.trees = c(100,500,1000)".       500 the best.slightly outperform 100 sometimes.
#                                                             100 best among the rest.
#                           best auc = 0.542
#       2) with 10000 rows: "interaction.depth =  c(1, 5)"     5 is better
#                           "n.trees = c(100,500,1000),"       1000 is the best
#                           "shrinkage = c(0.001),"
#                            best AUC = 0.6219176

# parameters optimizing:
gbmGrid <-  expand.grid(interaction.depth =  c(6,10),
                        n.trees = c(1000,2000),
                        shrinkage = c(0.001),
                        n.minobsinnode=10)

model_gbm <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                   method='gbm', 
                   trControl=objControl,  
                   metric = "ROC",
                   tuneGrid = gbmGrid)

plot(model_gbm)

# knn (caret) -------------------------------------------------------------------------------
model_knn <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                   method='knn', 
                   trControl=objControl, 
                   metric = "ROC",
                   tuneLength = 20)
