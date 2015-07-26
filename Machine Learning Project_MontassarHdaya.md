---
title: "Human Activity Recognition"
author: 'Montassar H''Daya : Montassar.hdaya@gmail.com'
date: "July 26, 2015"
output:
  pdf_document:
    toc: yes
  html_document:
    keep_md:
    - yes
    number_sections:
    - yes
    toc:
    - yes
---
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
```{r, echo = FALSE}
setwd("~/R/R1/8.Machine Learning/Machine Learning Project")
pmlTrain<-read.csv("pml-training.csv", header=T, na.strings=c("NA", "#DIV/0!"))
pmlTest<-read.csv("pml-testing.csv", header=T, na.string=c("NA", "#DIV/0!"))
```
Training data was partitioned and preprocessed using the code described below. In brief, all variables with at least one "NA" were excluded from the analysis. Variables related to time and user information were excluded for a total of 51 variables and 19622 class measurements. Same variables were mainteined in the test data set (Validation dataset) to be used for predicting the 20 test cases provided.
```{r,, echo = FALSE}
## NA exclusion for all available variables
noNApmlTrain<-pmlTrain[, apply(pmlTrain, 2, function(x) !any(is.na(x)))] 
dim(noNApmlTrain)
```

```{r,, echo = FALSE}
## variables with user information, time and undefined
cleanpmlTrain<-noNApmlTrain[,-c(1:8)]
dim(cleanpmlTrain)
```
```{r, echo = FALSE}
## 20 test cases provided clean info - Validation data set
cleanpmltest<-pmlTest[,names(cleanpmlTrain[,-52])]
dim(cleanpmltest)
```

## Data Partitioning and Prediction Process
The cleaned downloaded data set was subset in order to generate a test set independent from the 20 cases provided set. Partitioning was performed to obtain a 75% training set and a 25% test set.

## data cleaning
```{r, echo = FALSE}
library(caret)
inTrain<-createDataPartition(y=cleanpmlTrain$classe, p=0.75,list=F)
training<-cleanpmlTrain[inTrain,] 
test<-cleanpmlTrain[-inTrain,] 
```{r}

## Training and test set dimensions
```{r, echo = FALSE}
dim(training)
```

```{r, echo = FALSE}
dim(test)
```
# Data analysis and estimate error with cross-validation
Random forest trees were generated for the training dataset using cross-validation. Then the generated algorithm was examnined under the partitioned training set to examine the accuracy and estimated error of prediction. By using 51 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.2% with a 95% CI [0.989-0.994] was achieved accompanied by a Kappa value of 0.99.

```{r, echo = FALSE}
library(caret)
library(e1071)
library(rattle)
set.seed(13333)
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=training, method="rf", trControl=fitControl2, verbose=F)

```

```{r, echo = FALSE}
## Fitting mtry = 26 on full training set
predrf<-predict(rffit, newdata=test)
## Confusion Matrix and Statistics
confusionMatrix(predrf, test$classe)

```
```{r, echo = FALSE}
pred20<-predict(rffit, newdata=cleanpmltest)
```

## Output for the prediction of the 20 cases provided
```{r, echo = FALSE}
pred20
```
A boosting algorithm was also run to confirm and be able to compare predictions. Data is not shown but the boosting approach presented less accuracy (96%) (Data not shown). However, when the predictions for the 20 test cases were compared match was same for both ran algorimths.

```{r, echo = FALSE}
library (gbm)
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
gmbfit<-train(classe~.,data=training, method="gbm", trControl=fitControl2, verbose=F)
gmbfit$finalModel
class(gmbfit)
predgmb<-predict(gmbfit, newdata=test)
confusionMatrix(predgmb, test$classe)
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
```
# Generating Files to submit as answers for the Assignment:
Once, the predictions were obtained for the 20 test cases provided, the below shown script was used to obtain single text files to be uploaded to the courses web site to comply with the submission assigment. 20 out of 20 hits also confirmed the accuracy of the obtained models.


```{r, echo = FALSE}

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred20) 
pred20
```