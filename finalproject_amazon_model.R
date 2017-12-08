# SYS - Final Project
# Amazon fine food review
# https://www.kaggle.com/snap/amazon-fine-food-reviews

#setwd('/home/yingjie/Desktop/SYS_Final_Proj')
setwd('/Users/Pan/Google Drive/Data Science/SYS 6018')

library(caret)
library(MASS) #stepAIC
library(pROC) #ROC, AUC
library(e1071) #svm
library(caretEnsemble)

data<-read.csv("modeldata_3.csv")
############################################ -------------------------------------------
#               Data Preprocess            #
############################################ 

data<-data[,2:704] #drop the weird "X" column
colnames(data)[c(1,2,3,4)]<-c("id","help_int","score","summary_length","text_length")  #standardized column names
data$help_int<-factor(data$help_int)

nrow(data[data$help_int==1,]) #70887
nrow(data[data$help_int==0,]) #324083
all1<-which(data$help_int==1,arr.ind=TRUE)  #get index for helpful review
all0<-which(data$help_int==0,arr.ind=TRUE)  #get index for unhelpful review

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
trainrows.bal<-c(sample(all1,size=5000),sample(all0,size=5000)) #trainrows for balance sampling
trainSet <- data[trainrows,]
testSet <- data[-trainrows,]

trainSet.b <- data[trainrows.bal,] #balance train set
testSet.b <- data[-trainrows.bal,]

finalrows<-sample(1:nrow(data),size=100000) 
finalSet <- data[finalrows,]
finaltestSet<-data[-finalrows,]

# save outcome's name and predictors'names
outcomeName<-'help_int'
predictorsNames<-names(trainSet)[!names(trainSet) %in% c(outcomeName,'id')]

############################################ --------------------------------------------------
#           Models (unbalcanced)           #
############################################
# cross validation setting:
    # 2-fold cross valiadation.
    # return class probability.
objControl <- trainControl(method='cv', number=2, 
                           returnResamp='none', summaryFunction = twoClassSummary, 
                           classProbs = TRUE,verboseIter=FALSE,
                           allowParallel= TRUE)

# Logistic regression - step (it takes too long to run (>2days))-------------------------------
glm1 <- glm(as.factor(help_int)~.-id, data=data[trainrows,], family = binomial(link = 'logit'))
step <- stepAIC(glm1, direction="backward")
step$anova


# Logistic regression -------------------------------------------------------------------------
model_glm<-train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                 method='glm')
#summary(model_glm)
predictions <- predict(object=model_glm, trainSet[,predictorsNames])
auc <- roc(ifelse(trainSet[,outcomeName]=="X2",1,0), ifelse(predictions=="X2",1,0))
print(auc$auc) #AUC=0.5401

# Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")
glm_imp<-varImp(object=model_glm)$importance
glm_imp$var<-row.names(glm_imp)
glm_imp<-glm_imp[order(-glm_imp$Overall),]
glm_imp_var<-glm_imp[glm_imp$Overall>10,'var'] #There are 271 variables have importance over 10

# svm （best AUC = 0.502) -----------------------------------------------------------------------------------------
model_weights <- ifelse(trainSet$help_int == "X1",
                        (1/table(trainSet$help_int)[1]) * 0.5,
                        (1/table(trainSet$help_int)[2]) * 0.5)

### Non-Linear Support Vector Machines with Class Weights
svmGrid_w <-  expand.grid(cost= c(.25, .5, 1),
                          weight = model_weights)

model_svm_w <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                   method='svmRadialWeights', 
                   trControl=objControl,  
                   metric = "ROC",
                   tuneGrid = svmGrid_w)

### Linear Support Vector Machines with Class Weights
svmGrid_w_r <-  expand.grid(cost= c(.25, .5, 1),
                            sigma = .05,
                            weight = model_weights)

model_svm_w_r <- train(trainSet[,predictorsNames], trainSet[,outcomeName], 
                     method='svmLinearWeights', 
                     trControl=objControl,  
                     metric = "ROC",
                     tuneGrid = svmGrid_w)


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

# parameter tuning history: 
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

# [FINAL] gbm (boosting) -------------------------------------------------
gbmGrid <-  expand.grid(interaction.depth =  c(5),
                        n.trees = c(1000),
                        shrinkage = c(0.001),
                        n.minobsinnode=10)

FINAL_model_gbm <- train(finalSet[,predictorsNames], finalSet[,outcomeName], 
                   method='gbm', 
                   trControl=trainControl(method="cv",number=1, 
                                          returnResamp='none', 
                                          summaryFunction = twoClassSummary, 
                                          classProbs = TRUE),  
                   metric = "ROC",
                   tuneGrid = gbmGrid)
# ROC = 0.4962443

# knn (caret) best AUC = 0.6044192 ------------------------------------------------------------------------
# not able to run 10000. Thus, we used 5000.
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
# The higher the K the better AUC.

plot(model_knn)

# kmeans+svm (caret) best AUC = 0.5032037------------------------------------------------------------------------
kmeans_var<-predictorsNames[3:700] #use only term frequecy. 
cor(data$text_length,as.numeric(data$help_int))    #0.1549582. add in text_length in second layer.
cor(data$summary_length,as.numeric(data$help_int)) #0.05789282. not adding in since not much relation.

aucdf<-data.frame()
for (k in 2:20){
  # first layer model: kmeans
  cl<-kmeans(trainSet[,kmeans_var], centers=k)
  
  tempdf<-data.frame(trainSet$help_int,as.factor(cl$cluster))
  names(tempdf)<-c("help_int","cluster")
  aucrow<-c(k)
  
  # second layer model: svm
  for (r in c("radial","linear")){
    tempsvm<-svm(as.factor(help_int)~., data=tempdf, kernel=r)  
    print ("---------------------------------")
    print (paste("k = ",as.character(k)))
    print (paste("kernel = ",r))
    auc <- roc(ifelse(tempdf[,outcomeName]=="X2",1,0), ifelse(fitted(tempsvm)=="X2",1,0))
    aucrow<-c(aucrow,auc$auc[1])
  }
  aucdf<-rbind(aucdf,aucrow)
  }
colnames(aucdf)<-c("k","radial","linear")
aucdf
#1) with text_length
#     k    radial linear
# 1   2 0.5021358    0.5
# 2   3 0.5025178    0.5
# 3   4 0.5032037    0.5
# 4   5 0.5021749    0.5
# 5   6 0.5022357    0.5
# 6   7 0.5020750    0.5
# 7   8 0.5023572    0.5
# 8   9 0.5022357    0.5
# 9  10 0.5022357    0.5
# 10 11 0.5020143    0.5
# 11 12 0.5022964    0.5
# 12 13 0.5020750    0.5
# 13 14 0.5022964    0.5
# 14 15 0.5020143    0.5
# 15 16 0.5017929    0.5
# 16 17 0.5017929    0.5
# 17 18 0.5017929    0.5
# 18 19 0.5017929    0.5
# 19 20 0.5017929    0.5

#1) without text_length. useless :(
#     k radial linear
# 1   2    0.5    0.5
# 2   3    0.5    0.5
# 3   4    0.5    0.5
# 4   5    0.5    0.5
# 5   6    0.5    0.5
# 6   7    0.5    0.5
# 7   8    0.5    0.5
# 8   9    0.5    0.5
# 9  10    0.5    0.5
# 10 11    0.5    0.5
# 11 12    0.5    0.5
# 12 13    0.5    0.5
# 13 14    0.5    0.5
# 14 15    0.5    0.5
# 15 16    0.5    0.5
# 16 17    0.5    0.5
# 17 18    0.5    0.5
# 18 19    0.5    0.5
# 19 20    0.5    0.5


############################################ --------------------------------------------------
#           Models (balcanced)             #
############################################
#tune grid for gbm
gbmGrid.b <-  expand.grid(interaction.depth =  c(5),
                        n.trees = c(500,1000),
                        shrinkage = c(0.01,0.001),
                        n.minobsinnode=10)

# create submodels
models <- caretList(trainSet.b[,predictorsNames], trainSet.b[,outcomeName], 
                    metric = "ROC", 
                    trControl=objControl, 
                    tuneList = list(
                      knn.b=caretModelSpec(method='knn',tuneLength=5),
                      glm.b=caretModelSpec(method='glm',family='binomial'),
                      rpart.b = caretModelSpec(method='rpart'),
                      gbm.b=caretModelSpec(method='gbm',tuneGrid = gbmGrid.b)
                    ))
results <- resamples(models)
summary(results)
dotplot(results)
# Call:
#   summary.resamples(object = results)

# Models: knn.b, glm.b, rpart.b, gbm.b 
# Number of resamples: 2 

# ROC 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn.b   0.5722050 0.5765016 0.5807982 0.5807982 0.5850948 0.5893914    0
# glm.b   0.5900546 0.5913363 0.5926180 0.5926180 0.5938997 0.5951814    0
# rpart.b 0.5846698 0.5888747 0.5930797 0.5930797 0.5972846 0.6014896    0
# gbm.b   0.6363413 0.6369866 0.6376320 0.6376320 0.6382774 0.6389227    0

# Sens 
#           Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
# knn.b   0.5284  0.5348 0.5412 0.5412  0.5476 0.5540    0
# glm.b   0.5584  0.5633 0.5682 0.5682  0.5731 0.5780    0
# rpart.b 0.5036  0.5281 0.5526 0.5526  0.5771 0.6016    0
# gbm.b   0.6276  0.6357 0.6438 0.6438  0.6519 0.6600    0

# Spec 
# Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
# knn.b   0.5628  0.5650 0.5672 0.5672  0.5694 0.5716    0
# glm.b   0.5588  0.5650 0.5712 0.5712  0.5774 0.5836    0
# rpart.b 0.5540  0.5759 0.5978 0.5978  0.6197 0.6416    0
# gbm.b   0.5340  0.5437 0.5534 0.5534  0.5631 0.5728    0


############################################ --------------------------------------------------
# Models (with score variable & balanced)  #
############################################

## Results after adding score variable - slightly lower than before
# Call:
#   summary.resamples(object = results)
# 
# Models: knn.b, glm.b, rpart.b, gbm.b 
# Number of resamples: 2 
# 
# ROC 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn.b   0.5780099 0.5792950 0.5805802 0.5805802 0.5818653 0.5831504    0
# glm.b   0.5856276 0.5866640 0.5877004 0.5877004 0.5887367 0.5897731    0
# rpart.b 0.6015031 0.6016186 0.6017340 0.6017340 0.6018495 0.6019650    0
# gbm.b   0.6375743 0.6375781 0.6375820 0.6375820 0.6375858 0.6375896    0
# 
# Sens 
#           Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
# knn.b   0.5592  0.5628 0.5664 0.5664  0.5700 0.5736    0
# glm.b   0.5584  0.5600 0.5616 0.5616  0.5632 0.5648    0
# rpart.b 0.6060  0.6109 0.6158 0.6158  0.6207 0.6256    0
# gbm.b   0.6208  0.6249 0.6290 0.6290  0.6331 0.6372    0
# 
# Spec 
# Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
# knn.b   0.5520  0.5524 0.5528 0.5528  0.5532 0.5536    0
# glm.b   0.5660  0.5685 0.5710 0.5710  0.5735 0.5760    0
# rpart.b 0.5596  0.5618 0.5640 0.5640  0.5662 0.5684    0
# gbm.b   0.5708  0.5720 0.5732 0.5732  0.5744 0.5756    0

