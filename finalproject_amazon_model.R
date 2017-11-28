# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC
library(pROC) #ROC, AUC

data<-read.csv("J:/private/modeldata_2.csv")

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
print(auc$auc) #0.5401

# Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")
glm_imp<-varImp(object=model_glm)$importance
glm_imp$var<-row.names(glm_imp)
glm_imp<-glm_imp[order(-glm_imp$Overall),]
glm_imp_var<-glm_imp[glm_imp$Overall>10,'var'] #271


# gbm (boosting) (caret) best AUC= 0.622----------------------------------------------------------------------
# parameters optimizing:
gbmGrid <-  expand.grid(interaction.depth =  c(1,5,10),
                        n.trees = c(500,1000,1500,2000),
                        shrinkage = c(0.01,0.001),
                        n.minobsinnode=10)

model_gbm <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                   method='gbm', 
                   trControl=objControl,  
                   metric = "ROC",
                   tuneGrid = gbmGrid)
model_gbm

# history: 
#       1) with 10000 rows (without summary length and text length): 
#                           "interaction.depth =  c(1, 5)"     5 is better
#                           "n.trees = c(100,500,1000),"       1000 is the best
#                           "shrinkage = c(0.001),"
#                            best AUC = 0.6219176
#   2) with 10000 rows (with summary length and text length)
#       best AUC =  0.6221544
#       comment: increase ntree from 1000 to 2000 doesn't help a lot.
# shrinkage  interaction.depth  n.trees  ROC        Sens       Spec       
# 0.001       1                  500     0.6162054  1.0000000  0.000000000
# 0.001       1                 1000     0.6198955  1.0000000  0.000000000
# 0.001       1                 1500     0.6204239  1.0000000  0.000000000
# 0.001       1                 2000     0.6207190  1.0000000  0.000000000
# 0.001       5                  500     0.6207595  1.0000000  0.000000000
# ***0.001       5                 1000     0.6220750  1.0000000  0.000000000***
# 0.001       5                 1500     0.6217820  1.0000000  0.000000000
# ***0.001       5                 2000     0.6221544  1.0000000  0.000000000***
# 0.001      10                  500     0.6181787  1.0000000  0.000000000
# 0.001      10                 1000     0.6191999  1.0000000  0.000000000
# 0.001      10                 1500     0.6196583  1.0000000  0.000000000
# 0.001      10                 2000     0.6186000  0.9998785  0.000000000
# 0.010       1                  500     0.6205493  1.0000000  0.000000000
# 0.010       1                 1000     0.6181108  1.0000000  0.000000000
# 0.010       1                 1500     0.6150899  0.9996354  0.000000000
# 0.010       1                 2000     0.6132176  0.9985416  0.003386005
# 0.010       5                  500     0.6151345  0.9993923  0.001693002
# 0.010       5                 1000     0.6073571  0.9952601  0.010158014
# 0.010       5                 1500     0.6010847  0.9902771  0.019751693
# 0.010       5                 2000     0.5975893  0.9831065  0.030474041
# 0.010      10                  500     0.6091713  0.9970831  0.008465011
# 0.010      10                 1000     0.6000915  0.9885756  0.021444695
# 0.010      10                 1500     0.5932386  0.9782450  0.037810384
# 0.010      10                 2000     0.5891345  0.9725328  0.047968397

# knn (caret) ------------------------------------------------------------------------
# not able to run 10000
model_knn <- train(trainSet[c(1:5000),predictorsNames], trainSet[c(1:5000),outcomeName], 
                   method='knn', 
                   trControl=objControl, 
                   metric = "ROC",
                   tuneLength = 20)
# history: 
#       1) with 5000 rows (without summary length and text length): 
# k   ROC        Sens       Spec       
# 5  0.5373886  0.9149123  0.100207410
# 7  0.5475508  0.9423116  0.079921014
# 9  0.5440318  0.9665883  0.046525173
# 11  0.5524388  0.9826931  0.031017729
# 13  0.5535619  0.9882217  0.028631094
# 15  0.5554411  0.9932696  0.011927492
# 17  0.5483060  0.9959139  0.004770428
# 19  0.5554560  0.9978369  0.001193317
# 21  0.5543629  0.9990385  0.000000000
# 23  0.5536180  0.9997596  0.000000000
# 25  0.5534177  1.0000000  0.000000000
# 27  0.5538799  1.0000000  0.000000000
# 29  0.5517360  1.0000000  0.000000000
# ****31  0.5564034****  1.0000000  0.000000000
# 33  0.5538596  1.0000000  0.000000000
# 35  0.5497490  1.0000000  0.000000000
# 37  0.5471995  1.0000000  0.000000000
# 39  0.5495801  1.0000000  0.000000000
# 41  0.5522381  1.0000000  0.000000000
# 43  0.5563074  1.0000000  0.000000000

plot(model_knn)
# Stacking ----------------------------------------------------------------------------------
gbm_prob <- predict(object=model_glm, trainSet[,predictorsNames], type="prob")
