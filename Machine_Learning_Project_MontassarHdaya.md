# Human Activity Recognition
Montassar H'Daya : Montassar.hdaya@gmail.com  
July 26, 2015  
# Executive Summary

This detailed analysis has been performed to fulfill the requirements of the course project for the course Pratical Machine learning offered by the Johns Hopkins University on Data science specialization. 

Human Activity Recognition - HAR - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community (see picture below, that illustrates the increasing number of publications in HAR with wearable accelerometers), especially for the development of context-aware systems. There are many potential applications for HAR, like: elderly monitoring, life log systems for monitoring energy expenditure and for supporting weight-loss programs, and digital assistants for weight lifting exercises.

Human activity recognition research has traditionally focused on discriminating between different activities. However, the "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training (http://groupware.les.inf.puc-rio.br/har).

For the prediction of how welll individuals performed the assigned exercise six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

This report aims to use machine learning algoritmhs to predict the class of exercise the individuals was performing by using meaurements available from devices such as Jawbone Up, Nike FuelBand, and Fitbit.


#Data 
The data for this project come was obteined from http://groupware.les.inf.puc-rio.br/har. Two data set were available a training set and a test set for which 20 individuals without any classification for the class of exercise was available.

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


# Extract , transform and load the Data 

```r
setwd("~/R/R1/8.Machine Learning/Machine Learning Project")
pmlTrain<-read.csv("pml-training.csv", header=T, na.strings=c("NA", "#DIV/0!"))
pmlTest<-read.csv("pml-testing.csv", header=T, na.string=c("NA", "#DIV/0!"))
```
Training data was partitioned and preprocessed using the code described below. In brief, all variables with at least one "NA" were excluded from the analysis. Variables related to time and user information were excluded for a total of 51 variables and 19622 class measurements. Same variables were mainteined in the test data set (Validation dataset) to be used for predicting the 20 test cases provided.

```r
## NA exclusion for all available variables
noNApmlTrain<-pmlTrain[, apply(pmlTrain, 2, function(x) !any(is.na(x)))] 
dim(noNApmlTrain)
```

```
## [1] 19622    60
```


```r
## variables with user information, time and undefined
cleanpmlTrain<-noNApmlTrain[,-c(1:8)]
dim(cleanpmlTrain)
```

```
## [1] 19622    52
```

```r
## 20 test cases provided clean info - Validation data set
cleanpmltest<-pmlTest[,names(cleanpmlTrain[,-52])]
dim(cleanpmltest)
```

```
## [1] 20 51
```

## Data Partitioning and Prediction Process
The cleaned downloaded data set was subset in order to generate a test set independent from the 20 cases provided set. Partitioning was performed to obtain a 75% training set and a 25% test set.

## data cleaning

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain<-createDataPartition(y=cleanpmlTrain$classe, p=0.75,list=F)
training<-cleanpmlTrain[inTrain,] 
test<-cleanpmlTrain[-inTrain,] 
```

```r
## Training and test set dimensions
```

```r
dim(training)
```

```
## [1] 14718    52
```


```r
dim(test)
```

```
## [1] 4904   52
```
# Data analysis and estimate error with cross-validation
Random forest trees were generated for the training dataset using cross-validation. Then the generated algorithm was examnined under the partitioned training set to examine the accuracy and estimated error of prediction. By using 51 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.2% with a 95% CI [0.989-0.994] was achieved accompanied by a Kappa value of 0.99.


```r
library(caret)
library(e1071)
library(rattle)
```

```
## Loading required package: RGtk2
## Rattle: A free graphical interface for data mining with R.
## Version 3.5.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
set.seed(13333)
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=training, method="rf", trControl=fitControl2, verbose=F)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 26 on full training set
```


```r
## Fitting mtry = 26 on full training set
predrf<-predict(rffit, newdata=test)
## Confusion Matrix and Statistics
confusionMatrix(predrf, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  942    6    0    0
##          C    0    2  845    6    2
##          D    0    0    4  798    2
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9945         
##                  95% CI : (0.992, 0.9964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.993          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9883   0.9925   0.9956
## Specificity            0.9986   0.9985   0.9975   0.9985   1.0000
## Pos Pred Value         0.9964   0.9937   0.9883   0.9925   1.0000
## Neg Pred Value         1.0000   0.9982   0.9975   0.9985   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1723   0.1627   0.1829
## Detection Prevalence   0.2855   0.1933   0.1743   0.1639   0.1829
## Balanced Accuracy      0.9993   0.9956   0.9929   0.9955   0.9978
```

```r
pred20<-predict(rffit, newdata=cleanpmltest)
```

## Output for the prediction of the 20 cases provided

```r
pred20
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
A boosting algorithm was also run to confirm and be able to compare predictions. Data is not shown but the boosting approach presented less accuracy (96%) (Data not shown). However, when the predictions for the 20 test cases were compared match was same for both ran algorimths.


```r
library (gbm)
```

```
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
gmbfit<-train(classe~.,data=training, method="gbm", trControl=fitControl2, verbose=F)
```

```
## Loading required package: plyr
```

```
## + Fold1: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold1: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold1: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## Aggregating results
## Selecting tuning parameters
## Fitting n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10 on full training set
```

```r
gmbfit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 51 predictors of which 42 had non-zero influence.
```

```r
class(gmbfit)
```

```
## [1] "train"         "train.formula"
```

```r
predgmb<-predict(gmbfit, newdata=test)
confusionMatrix(predgmb, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1379   28    0    0    2
##          B    9  890   41    3    9
##          C    5   27  803   33   11
##          D    2    1    9  764   11
##          E    0    3    2    4  868
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9592          
##                  95% CI : (0.9533, 0.9646)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9484          
##  Mcnemar's Test P-Value : 4.338e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9885   0.9378   0.9392   0.9502   0.9634
## Specificity            0.9915   0.9843   0.9812   0.9944   0.9978
## Pos Pred Value         0.9787   0.9349   0.9135   0.9708   0.9897
## Neg Pred Value         0.9954   0.9851   0.9871   0.9903   0.9918
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2812   0.1815   0.1637   0.1558   0.1770
## Detection Prevalence   0.2873   0.1941   0.1792   0.1605   0.1788
## Balanced Accuracy      0.9900   0.9611   0.9602   0.9723   0.9806
```

```r
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4134   61    0    1    1
##          B   31 2725   60    6   14
##          C   14   58 2480   61   22
##          D    5    0   24 2327   31
##          E    1    4    3   17 2638
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9719          
##                  95% CI : (0.9691, 0.9745)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9644          
##  Mcnemar's Test P-Value : 1.351e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9878   0.9568   0.9661   0.9648   0.9749
## Specificity            0.9940   0.9906   0.9872   0.9951   0.9979
## Pos Pred Value         0.9850   0.9609   0.9412   0.9749   0.9906
## Neg Pred Value         0.9952   0.9896   0.9928   0.9931   0.9944
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2809   0.1851   0.1685   0.1581   0.1792
## Detection Prevalence   0.2852   0.1927   0.1790   0.1622   0.1809
## Balanced Accuracy      0.9909   0.9737   0.9767   0.9799   0.9864
```

```r
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4134   61    0    1    1
##          B   31 2725   60    6   14
##          C   14   58 2480   61   22
##          D    5    0   24 2327   31
##          E    1    4    3   17 2638
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9719          
##                  95% CI : (0.9691, 0.9745)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9644          
##  Mcnemar's Test P-Value : 1.351e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9878   0.9568   0.9661   0.9648   0.9749
## Specificity            0.9940   0.9906   0.9872   0.9951   0.9979
## Pos Pred Value         0.9850   0.9609   0.9412   0.9749   0.9906
## Neg Pred Value         0.9952   0.9896   0.9928   0.9931   0.9944
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2809   0.1851   0.1685   0.1581   0.1792
## Detection Prevalence   0.2852   0.1927   0.1790   0.1622   0.1809
## Balanced Accuracy      0.9909   0.9737   0.9767   0.9799   0.9864
```
# Generating Files to submit as answers for the Assignment:
Once, the predictions were obtained for the 20 test cases provided, the below shown script was used to obtain single text files to be uploaded to the courses web site to comply with the submission assigment. 20 out of 20 hits also confirmed the accuracy of the obtained models.



```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred20) 
```
