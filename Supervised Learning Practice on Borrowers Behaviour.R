#=======================================================================================================
## Supervised Learning Practice on Borrowers Behaviour
#=======================================================================================================

#=======================================================================================================
## Part 1. Preparation & Variables Treatment
#=======================================================================================================

# set working directory
setwd()

# clear workspace
rm(list=ls())

## Note:
# We split our whole global enviroment into "Classifier_SVM" and "Classifier_ANN" based on the analysis approaches.
# Please load "Classifier_SVM.RData" at first for "Part 1: Variables Treatment" and "Part 2: SVM analysis".

# load library
library(e1071) # SVM
library(caret)
library(nnet) # neural networks
library(monmlp) # Monotone Multi-Layer Perceptron Neural Network
library(randomForest) #Random Forest
library(plotly) # Visualization
library(corrplot) # correlation matrix

## load data set
# Beside target variable, we only have two categorial variables. Transfer them to factor.
heloc <- read.csv("heloc_dataset_v1.csv")
heloc$MaxDelqEver <- as.factor(heloc$MaxDelqEver)
heloc$MaxDelq2PublicRecLast12M <- as.factor(heloc$MaxDelq2PublicRecLast12M)

## Data Structure
str(heloc)
summary(heloc)

## Feature selection
set.seed(123)
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 10,
                      repeats = 3,
                      verbose = FALSE)

VI.rf <- rfe(x = train[,-1], 
                y = train$RiskPerformance,
                rfeControl = control)

VI.rf 

## Note: 
# If you can't see graph directly, that becasue the library version dismatches between your environment and 
# some functions that library plotly used. We recommend you use "Export" - "Save as web page" in "Viewer" panel.
# Then you can see plot (in html format) in the browser.

## Boxplots and outliers
box1 <- plot_ly(y= heloc[,2], type = "box",name ="ExternalRiskEsitmate")
box1 <- box1 %>% add_trace(y= heloc[,3], type = "box",name ="MSinceOldestTradeOpen")
box1 <- box1 %>% add_trace(y= heloc[,4], type = "box",name ="MSinceMostRecentTradeOpen")
box1 <- box1 %>% add_trace(y= heloc[,5], type = "box",name ="AverageMInFile")
box1 <- box1 %>% add_trace(y= heloc[,6], type = "box",name ="NumSatisfactoryTrades")
box1 <- box1 %>% add_trace(y= heloc[,19], type = "box",name ="NetFractionRevolvingBurden")
box1 <- box1 %>% add_trace(y= heloc[,20], type = "box",name ="NetFractionInstallBurden ")
box1 <- box1 %>% layout(title = "Boxplots of all numerical variables ")
box1

box2 <- plot_ly(y= heloc[,9], type = "box",name ="PercentTradesNeverDelq")
box2 <- box2 %>% add_trace(y= heloc[,10], type = "box",name ="MSinceMostRecentDelq")
box2 <- box2 %>% add_trace(y= heloc[,13], type = "box",name ="NumTotalTrades")
box2 <- box2 %>% add_trace(y= heloc[,15], type = "box",name ="PercentInstallTrades")
box2 <- box2 %>% add_trace(y= heloc[,17], type = "box",name ="NumInqLast6M")
box2 <- box2 %>% add_trace(y= heloc[,24], type = "box",name ="PercentTradesWBalance ")
box2 <- box2 %>% layout(title = "Boxplots of all numerical variables ")
box2

box3 <- plot_ly(y= heloc[,18], type = "box",name ="NumInqLast6Mexcl7days")
box3 <- box3 %>% add_trace(y= heloc[,7], type = "box",name ="NumTrades60Ever2DerogPubRec")
box3 <- box3 %>% add_trace(y= heloc[,8], type = "box",name ="NumTrades90Ever2DerogPubRec")
box3 <- box3 %>% add_trace(y= heloc[,14], type = "box",name ="NumTradesOpeninLast12M")
box3 <- box3 %>% add_trace(y= heloc[,16], type = "box",name ="MSinceMostRecentInqexcl7days ")
box3 <- box3 %>% add_trace(y= heloc[,21], type = "box",name ="NumRevolvingTradesWBalance")
box3 <- box3 %>% add_trace(y= heloc[,22], type = "box",name ="NumInstallTradesWBalance")
box3 <- box3 %>% add_trace(y= heloc[,23], type = "box",name ="NumBank2NatlTradesWHighUtilization")
box3 <- box3 %>% layout(title = "Boxplots of all numerical variables ")
box3

## Correlation
heloc.nofactor <- heloc[,-c(1,11,12)]
cor <- cor(heloc.nofactor)
cor
findCorrelation(cor, cutoff=.75, names=TRUE)
cor_plot <- corrplot(cor,method = "circle",tl.cex = 0.6)

#=======================================================================================================
## Part 2. SVM
#=======================================================================================================

## 1. data structure
nrow(heloc[!complete.cases(heloc),])       # no missing value
barplot(table(heloc$RiskPerformance))      
prop.table(table(heloc$RiskPerformance))    # "Bad" is slightly higher than "Good"

## 2. Dummy Conversion
df1 <- dummyVars(~MaxDelq2PublicRecLast12M+MaxDelqEver, data=heloc,
                 sep="_", fullRank = TRUE)
df2 <- predict(df1, heloc)
heloc.dum <- data.frame(heloc[,-c(11,12)], df2)
str(heloc.dum)

## 3. Split train & test data set
set.seed(123)
samp <- createDataPartition(heloc.dum$RiskPerformance, p=.80, list=FALSE)
train = heloc.dum[samp, ] 
test = heloc.dum[-samp, ]

## 4. Basic SVM to find model with best training performance

# 4.1). Linear Kernel
#set.seed(123)
svm.linear <- svm(RiskPerformance~., 
               data=train, 
               method="C-classification", 
               kernel="linear", 
               scale=F)
# Training Performance
svm.linear.train <- predict(svm.linear, train[,-1], type="class")
svm.linear.train.acc <- confusionMatrix(svm.linear.train, train$RiskPerformance, mode="prec_recall")
svm.linear.train.acc
plot(x = svm.linear, data = train, formula = ExternalRiskEstimate ~ AverageMInFile)

# 4.2). Radial Kernel
set.seed(123)
svm.radial <- svm(RiskPerformance~., 
                data=train, 
                method="C-classification", 
                kernel="radial", 
                scale=TRUE)
# Training Performance
svm.radial.train <- predict(svm.radial, train[,-1], type="class")
svm.radial.train.acc <- confusionMatrix(svm.radial.train, train$RiskPerformance, mode="prec_recall")
svm.radial.train.acc

# 4.3). Polynomial Kernel
set.seed(123)
svm.polynomial <- svm(RiskPerformance~., 
                  data=train, 
                  method="C-classification", 
                  kernel="polynomial", 
                  scale=TRUE)
# Training Performance
svm.polynomial.train <- predict(svm.polynomial, train[,-1], type="class")
svm.polynomial.train.acc <- confusionMatrix(svm.polynomial.train, train$RiskPerformance, mode="prec_recall")
svm.polynomial.train.acc

# 4.4). Sigmoid Kernel
set.seed(123)
svm.sigmoid <- svm(RiskPerformance~., 
                      data=train, 
                      method="C-classification", 
                      kernel="sigmoid", 
                      scale=TRUE)
# Training Performance
svm.sigmoid.train <- predict(svm.sigmoid, train[,-1], type="class")
svm.sigmoid.train.acc <- confusionMatrix(svm.sigmoid.train, train$RiskPerformance, mode="prec_recall")
svm.sigmoid.train.acc

## 5. Comparing Accuracy Across Kernels
r1 <- rbind(Linear=svm.linear.train.acc$overall[1], 
            Radial=svm.radial.train.acc$overall[1],
            Polynomial=svm.polynomial.train.acc$overall[1], 
            Sigmoid=svm.sigmoid.train.acc$overall[1])
r2 <- rbind(Linear=svm.linear.train.acc$byClass[5:7], 
            Radial=svm.radial.train.acc$byClass[5:7],
            Polynomial=svm.polynomial.train.acc$byClass[5:7], 
            Sigmoid=svm.sigmoid.train.acc$byClass[5:7])
cbind(r1,r2)

## 6. Hyperparameter Tuning
# Based on previous results that Radial & Polynomial kernel has best train performance

# 6.1). Radial Kernel
tRadial <- tune(svm, RiskPerformance~., data=train, 
             tunecontrol=tune.control(sampling = "cross"), 
             kernel="radial", scale = TRUE,
             ranges = list(gamma = 2^(-3:2), cost = 2^(-5:10)))
summary(tRadial)
tRadial$best.parameters

# Choose best train model performance
inpred <- predict(tRadial$best.model, train)
confusionMatrix(inpred, train$RiskPerformance, mode="prec_recall")

# Apply best train model to test data
outpred <- predict(tRadial$best.model, test)
confusionMatrix(outpred, test$RiskPerformance, mode="prec_recall")

# 6.2). Polynomial Kernel
tunecontrol <- tune.control(nrepeat = 3,
                            sampling = "cross")

ranges <- list(gamma = c(0.001, 0.01, 0.1),
               cost = c(1, 5),
               degree = c(3,4,5))

tPoly <- tune(svm, RiskPerformance~., data=train, 
              tunecontrol = tunecontrol, 
              kernel = "polynomial", 
              scale = TRUE,
              ranges = ranges)

summary(tPoly)
tPoly$best.parameters

# Choose best train model performance
inpred2 <- predict(tPoly$best.model, train)
confusionMatrix(inpred2, train$RiskPerformance, mode="prec_recall")

# Apply best train model to test data
outpred2 <- predict(tPoly$best.model, test)
confusionMatrix(outpred2, test$RiskPerformance, mode="prec_recall")

save.image("Classifier_SVM.RData")

#=======================================================================================================
## Part 3. ANN
#=======================================================================================================

## clear workspace
rm(list=ls())

## Note:
# We split our whole global enviroment into "Classifier_SVM" and "Classifier_ANN" based on the analysis approaches.
# Please load "Classifier_ANN.RData" at first for "Part 3. ANN" 
# and "Part 4. Monotone Multi-Layer Perceptron Neural Network"

## load data set
# Beside target variable, we only have two categorial variables. Let's transfer them to factor.
heloc <- read.csv("heloc_dataset_v1.csv")
# Haven't Change RiskPerformance to 0/1
heloc$RiskPerformance <- as.factor(heloc$RiskPerformance)
heloc$MaxDelqEver <- as.factor(heloc$MaxDelqEver)
heloc$MaxDelq2PublicRecLast12M <- as.factor(heloc$MaxDelq2PublicRecLast12M)

## data structure
str(heloc)
summary(heloc)
nrow(heloc[!complete.cases(heloc),])       # no missing value
barplot(table(heloc$RiskPerformance))      # Bascially Balanced Data
prop.table(table(heloc$RiskPerformance))    # "Bad" is slightly higher than "Good"

## Dummy Conversion
df1 <- dummyVars(~MaxDelq2PublicRecLast12M+MaxDelqEver, data=heloc,
                 sep="_", fullRank = TRUE)
df2 <- predict(df1, heloc)
heloc.dum <- data.frame(heloc[,-c(11,12)], df2)
str(heloc.dum)

## Rescaling
df3 <- preProcess(heloc.dum, method="range")
df4 <- predict(df3, heloc.dum)
summary(df4)

## Training & Testing
set.seed(123)
samp <- createDataPartition(df4$RiskPerformance, p=.80, list=FALSE)
train = df4[samp, ] 
test = df4[-samp, ]

## Base Line
nnmod <- nnet(RiskPerformance~., data=train, size=8, trace=FALSE)

ann.base.train <- predict(nnmod, train[,-1], type="class")

confusionMatrix(factor(ann.base.train), 
                train$RiskPerformance, 
                mode="prec_recall", positive = "Bad")

ann.base.test <- predict(nnmod, test[,-1], type = "class")
confusionMatrix(factor(ann.base.test), 
                test$RiskPerformance, 
                mode="prec_recall", positive = "Bad")

## Hyperparameter Tuning
grids = expand.grid(size = seq(from = 1, to = 8, by = 1),
                    decay = seq(from = 0.1, to = 0.5, by = 0.1))

ctrl <- trainControl(method="repeatedcv",
                     number = 10,
                     repeats=3,
                     search="grid")

set.seed(123)
ann.hyper.train <- train(RiskPerformance~ ., data = train, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids,
                verbose=FALSE)

ann.hyper.train
plot(ann.hyper.train)
confusionMatrix(ann.hyper.train)

# Training Performance
inpreds <- predict(ann.hyper.train, newdata=train)
confusionMatrix(inpreds, train$RiskPerformance, mode="prec_recall")

# Testing Performance
outpreds <- predict(ann.hyper.train, newdata=test)
confusionMatrix(outpreds, test$RiskPerformance, mode="prec_recall")

#=======================================================================================================
## Part 4. Monotone Multi-Layer Perceptron Neural Network
#=======================================================================================================

## 1. 3D graphs for monotone feature
fig <- plot_ly(heloc, x = ~MSinceOldestTradeOpen, y = ~PercentTradesNeverDelq, z = ~ExternalRiskEstimate, 
               color = ~RiskPerformance, 
               opacity = 0.3,  colors = c('#636EFA', '#FECB52')) 

fig <- fig %>% add_markers()

fig <- fig %>% layout(
  scene = list(
    xaxis = list(
      spikecolor = '#a009b5',
      spikesides = FALSE,
      spikethickness = 10),
    yaxis = list(
      spikecolor = '#a009b5',
      spikesides = FALSE,
      spikethickness = 10),
    zaxis = list(
      spikecolor = '#a009b5',
      spikethickness = 10)))

fig

## 2. By using caret with library monmlp's API
monmlp.ModelInfo <- getModelInfo(model = "monmlp", regex = FALSE)[[1]]
# Available Model components
names(monmlp.ModelInfo)
# Available Model Hyperparameters
modelLookup('monmlp')    # We can use two hyperparameters: hidden units and n.ensemble

# Filter-based variable importance
filterVarImp(train.Data,train.Classes) 

monmlp.grids = expand.grid(hidden1 = seq(from = 1, to = 8, by = 1),
                           n.ensemble = seq(from = 1, to = 8, by =1))

monmlp.ctrl <- trainControl(method="repeatedcv",
                     number = 10,
                     repeats=3,
                     search="grid")
# create x and y 
train.Data <- train[,-1]
train.Classes <- train[,1]

# fit monmlp model
monmlp.train <- train(x = train.Data, y = train.Classes, 
                      method = "monmlp", iter.max = 500, monotone = c(6,7,11,14:17,20),
                      tuneLength = 10, 
                      trControl = monmlp.ctrl, tuneGrid = monmlp.grids,
                      verbose=FALSE)

# After tuning, hidden1 = 1, ensemble = 7
monmlp.train

# Training Performance
monmlp.inpreds <- predict(monmlp.train, newdata=train)
confusionMatrix(monmlp.inpreds, train$RiskPerformance, mode="prec_recall")

# Testing Performance
monmlp.outpreds <- predict(monmlp.train, newdata=test)
confusionMatrix(monmlp.outpreds, test$RiskPerformance, mode="prec_recall")

## 3. By using monmlp directly

#train.Data <- as.matrix(train.Data)
#train.Classes <- as.matrix(train.Classes)
#train.Classes[train.Classes=="Bad"] <- 1
#train.Classes[train.Classes=="Good"] <- 0
#train.Classes <- as.numeric(train.Classes)
#train.Classes <- as.matrix(train.Classes)

# mon.fit <- monmlp.fit(x = train.Data, y = train.Classes, 
                      #To = tansig, To.prime = tansig.prime,
                      #hidden1 = 8, monotone = c(6,7,11,14:17,20),
                      #scale = F,
                      #n.ensemble = 15, bag = TRUE, iter.max = 500)

#mon.pred <- attr(mon.fit, "y.pred")
#mon.pred <- mon.pred[,1]
#mon.pred <- monmlp.predict(x = as.matrix(train.Data), weights = mon.fit)
#mon.sigmoid <- sigmoid(mon.pred)

#summary(trp.p.mon)

#test.b <- sigmoid(b)
#test.b[test.b <= 0.5] <- 0
#test.b[test.b > 0.5] <- 1
#c <- test.b[,1]
#c <- as.factor(c)
#levels(c) <- c("Bad", "Good")

#confusionMatrix(c, train$RiskPerformance, mode="prec_recall")

## The analysis of directly using monmlp still has someplace needed to adjust.

save.image("Classifier_ANN.RData")

