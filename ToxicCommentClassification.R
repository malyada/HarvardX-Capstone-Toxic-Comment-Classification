if (!require(tidyverse)) install.packages('tidyverse')
if (!require(tidytext)) install.packages('tidytext')
if (!require(DT)) install.packages('DT')
if (!require(tm)) install.packages('tm')
if (!require(SnowballC)) install.packages('SnowballC')
if (!require(caret)) install.packages('caret')
if (!require(Metrics)) install.packages('Metrics')
if (!require(pROC)) install.packages('pROC')
if (!require(MLmetrics)) install.packages('MLmetrics')
if (!require(e1071)) install.packages('e1071')
if (!require(xgboost)) install.packages('xgboost')
if (!require(wordcloud)) install.packages('wordcloud')
if (!require(readr)) install.packages('readr')

library(tidyverse)
library(tidytext)
library(DT)
library(stringr)
library(tm)
library(SnowballC)
library(caret)
library(Metrics)
library(pROC)
library(MLmetrics)
library(e1071)
library(xgboost)
library(wordcloud)
library(readr)

#Note: For running the code in less time, uncomment the lines as mentioned in the middle of the code
#which allows subsetting the data appropriately. Then you can be running the model on a lesser data.
  
#For downloading the kaggle dataset, it needs user to be enrolled in the competetion in kaggle
#and accept its rules. This might be cumbersome when some one new decides to run the code. So
#will be downloading the data via github link

testpath <- "https://raw.githubusercontent.com/malyada/toxic_comment_classification/main/test.csv"
trainpath <- "https://raw.githubusercontent.com/malyada/toxic_comment_classification/main/train.csv"
sample_submissionpath <- "https://raw.githubusercontent.com/malyada/toxic_comment_classification/main/sample_submission.csv"
test_labelspath <- "https://raw.githubusercontent.com/malyada/toxic_comment_classification/main/test_labels.csv"

test <- read_csv(testpath)
train <- read_csv(trainpath)
sample_submission <- read_csv(sample_submissionpath)
test_labels <- read_csv(test_labelspath)

#cleaning the data
#removing puncutations from the text
train$comment_text=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", 
                        train$comment_text)
test$comment_text=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", 
                       test$comment_text)
#removing texts which are NA      
train <- train[!(is.na(train$comment_text)), ]
test <- test[!(is.na(test$comment_text)), ]
#looking at the data disribution among the classes (the data distribution is highly skewed)
cat('no. of observations:',(dim(train)[1]))
cat('no. of toxic observations:', sum(train$toxic == 1))
cat('no. of severe_toxic observations:', sum(train$severe_toxic == 1))
cat('no. of obscene observations:', sum(train$obscene == 1))
cat('no. of threat observations:', sum(train$threat == 1))
cat('no. of insult observations:', sum(train$insult == 1))
cat('no. of identity_hate observations:', sum(train$identity_hate == 1))
cat('no category observations:', sum((train$toxic == 0) & (train$severe_toxic == 0) & (train$obscene == 0) & 
            (train$threat == 0) & (train$insult == 0) & (train$identity_hate == 0)))

#lets look at the most popular words
head(train %>%
       unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
       mutate(word <- wordStem(word)) %>%
       anti_join(stop_words) %>%
       count(word, sort = T), 15)
#this seems fair as 90% of the sentences are good comments and do not belong to any categories. 
#lets get the data into the convenient tfidf per word per sentence format to make further analysis of the data per class

#looking at the wordcloud for each class to glace at the most frequent words.
#Note: This is a multilabel classification, each data point can belong to more than one labels
set.seed(1234) # for reproducibility 
#toxic word cloud
tmp <- train %>%
  filter(toxic == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))

#severe_toxic word cloud
tmp <- train %>%
  filter(severe_toxic == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))

#obscene word cloud
tmp <- train %>%
  filter(obscene == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))

#threat word cloud
tmp <- train %>%
  filter(threat == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))

#insult word cloud
tmp <- train %>%
  filter(insult == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))

#identity_hate word cloud
tmp <- train %>%
  filter(identity_hate == 1) %>%
  unnest_tokens(output = 'word', token = 'words', input = comment_text) %>%
  mutate(word <- wordStem(word)) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)

wordcloud(words = tmp$word, freq = tmp$n, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,            
          colors=brewer.pal(8, "Dark2"), scale=c(4, 0.75))
#we observe that there are many high frequency words which belong to more than one class.

#creating a document term matrix with TfIdf for dataset and datasetTest
train$comment_text = iconv(train$comment_text, 'UTF-8', 'ASCII')
train$comment_text=str_replace_all(train$comment_text,"[^[:graph:]]", " ") 

corpus = VCorpus(VectorSource(train$comment_text))

corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

dtm_train = DocumentTermMatrix(corpus, control = list(weightning = function(x) weightTfIdf(x, normalize = F)))
#it would crash unless below step is done 12gb ram is insufficient if not done
dtm_train = removeSparseTerms(dtm_train, 0.99)
dataset = as.data.frame(as.matrix(dtm_train))
dataset$toxic = NULL
dataset$severe_toxic = NULL
dataset$obscene = NULL
dataset$threat = NULL
dataset$insult = NULL
dataset$identity_hate = NULL

test$comment_text = iconv(test$comment_text, 'UTF-8', 'ASCII')
test$comment_text=str_replace_all(test$comment_text,"[^[:graph:]]", " ") 

corpus = VCorpus(VectorSource(test$comment_text))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)


dtm = DocumentTermMatrix(corpus, control = list(weightning = function(x) weightTfIdf(x, normalize = F)))
dtm = removeSparseTerms(dtm, 0.99)
datasetTest = as.data.frame(as.matrix(dtm))

#making the test and the train data set have the same columns ensuring no data leakage
#test matrix will have the word columns of the train, the train will not worry about the columns
#in the test, which is what happens in general.
matrix_column_names <- colnames(dataset)
intersectnames <- intersect(colnames(dataset),colnames(datasetTest))
datasetTest = datasetTest[ , (colnames(datasetTest) %in% intersectnames)]
trainminustestcolms <- matrix_column_names[!(matrix_column_names %in% intersectnames)]
for (s in trainminustestcolms) {
  datasetTest[, s] <- 0
}

#removing unnecessary variables
rm(corpus)
rm(dtm)
rm(dtm_train)
rm(tmp)
rm(intersectnames)
rm(matrix_column_names)
rm(s)
rm(sample_submissionpath)
rm(test_labelspath)
rm(testpath)
rm(trainminustestcolms)

#Using the linear kernel caused the system to crash, so we are positive that the data 
#is not linearly separated
#Also the classes are highly skewed. no label data is 90% of the training data, the largest
#class has 10% number of datapoints and the smallest class has .003% of the datapoints.

#among the classification algorithms present, one class classification svm radial kernel
#looks like a good idea to combact with the skewnedd and the nonlinearty.
#radial kernels can figure out complex patterns in the data. one class svm classifies

#uncomment below 3 lines to run the code with less data
#datasetTest <- datasetTest[1:100, ]
#test_labels <- test_labels[1:100, ]
#sample_submission <- sample_submission[1:100, ]

#Let us make 6 one class classifier models for all the 6 classes
#toxic
dataset2 = dataset 
dataset2$toxic = factor(train$toxic)
dataset2 <- dataset2 %>% filter(toxic == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-toxic)), y = dataset2$toxic, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionstoxic = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#severe_toxic
dataset2 = dataset 
dataset2$severe_toxic = factor(train$severe_toxic)
dataset2 <- dataset2 %>% filter(severe_toxic == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-severe_toxic)), y = dataset2$severe_toxic, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionssevere_toxic = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#obscene
dataset2 = dataset 
dataset2$obscene = factor(train$obscene)
dataset2 <- dataset2 %>% filter(obscene == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-obscene)), y = dataset2$obscene, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionsobscene = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#threat
dataset2 = dataset 
dataset2$threat = factor(train$threat)
dataset2 <- dataset2 %>% filter(threat == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-threat)), y = dataset2$threat, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionsthreat = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#insult
dataset2 = dataset 
dataset2$insult = factor(train$insult)
dataset2 <- dataset2 %>% filter(insult == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-insult)), y = dataset2$insult, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionsinsult = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#identity_hate
dataset2 = dataset 
dataset2$identity_hate = factor(train$identity_hate)
dataset2 <- dataset2 %>% filter(identity_hate == 1)
#uncomment below line to run on small data
#dataset2 <- dataset2[1:100, ]
tune_out <- 
  tune.svm(x = as.matrix(dataset2 %>% select(-identity_hate)), y = dataset2$identity_hate, 
           type = 'one-classification', 
           kernel = "radial", 
           #can specify more options at nu and gamma for hyper parameter optimization
           #doing this to speed up running the code for the evaluator, else it might take 
           #a very very long time
           nu = 0.1, #gamma will be default, 1/data dimension
           cross = 0) #to perform k fold cross validation, assign cross = k > 0
model <- tune_out$best.model

print(summary(model))

predictionsidentity_hate = ifelse(predict(model,datasetTest) == TRUE, 1, 0)

#the test_labels are the solutions given by kaggle, the -1 indicates they are not relavent examples
#and are not included in he scoring. such examples are added in kaggle to prevent malpractice
#so removing the datapoints with -1 in their labels
submission <- sample_submission
submission$toxic = predictionstoxic
submission$severe_toxic = predictionssevere_toxic
submission$obscene = predictionsobscene
submission$threat = predictionsthreat
submission$insult = predictionsinsult
submission$identity_hate = predictionsidentity_hate

y_svm <- submission[(test_labels$toxic != -1 & test_labels$severe_toxic != -1 & test_labels$obscene != -1
                     & test_labels$threat != -1 & test_labels$insult != -1 & test_labels$identity_hate != -1), ]

y_act <- test_labels[(test_labels$toxic != -1 & test_labels$severe_toxic != -1 & test_labels$obscene != -1
                      & test_labels$threat != -1 & test_labels$insult != -1 & test_labels$identity_hate != -1), ]

# f1 score metrics for onesvm
#doing this to be able to calculate f1 score even if all the y_actuals or y_predicted belong 
#to one class. If not done the recall may become 0/0 and this throws an error. 
#to avoid this, in such cases, we change the class one one of the observations.
if(length(unique(y_act$toxic)) == 1) {
  y_act$toxic[1] = (1 - y_act$toxic[1])
}
if(length(unique(y_act$severe_toxic)) == 1) {
  y_act$severe_toxic[1] = (1 - y_act$severe_toxic[1])
}
if(length(unique(y_act$threat)) == 1) {
  y_act$threat[1] = (1 - y_act$threat[1])
}
if(length(unique(y_act$obscene)) == 1) {
  y_act$Obscene[1] = (1 - y_act$obscene[1])
}
if(length(unique(y_act$insult)) == 1) {
  y_act$insult[1] = (1 - y_act$insult[1])
}
if(length(unique(y_act$identity_hate)) == 1) {
  y_act$identity_hate[1] = (1 - y_act$identity_hate[1])
}

if(length(unique(y_svm$toxic)) == 1) {
  y_svm$toxic[1] = (1 - y_svm$toxic[1])
}
if(length(unique(y_svm$severe_toxic)) == 1) {
  y_svm$severe_toxic[1] = (1 - y_svm$severe_toxic[1])
}
if(length(unique(y_svm$threat)) == 1) {
  y_svm$threat[1] = (1 - y_svm$threat[1])
}
if(length(unique(y_svm$obscene)) == 1) {
  y_svm$Obscene[1] = (1 - y_svm$obscene[1])
}
if(length(unique(y_svm$insult)) == 1) {
  y_svm$insult[1] = (1 - y_svm$insult[1])
}
if(length(unique(y_svm$identity_hate)) == 1) {
  y_svm$identity_hate[1] = (1 - y_svm$identity_hate[1])
}
#onesvm
toxic <- F1_Score(y_act$toxic, y_svm$toxic)
severe_toxic <- F1_Score(y_act$severe_toxic, y_svm$severe_toxic)
obscene <- F1_Score(y_act$obscene, y_svm$obscene)
threat <- F1_Score(y_act$threat, y_svm$threat)
insult <- F1_Score(y_act$insult, y_svm$insult)
identity_hate <- F1_Score(y_act$identity_hate, y_svm$identity_hate)
#macroaverage of f1
onesvmf1 <- print((toxic + severe_toxic + obscene + threat + insult + identity_hate)/6) 
cat("f1 score of one class svm at nu = 0.1 :", onesvmf1)

#using one class svm we got an okish f1 score.

#as experimented in colab notebook, with larger value of nu, the f1 score increased.
#with nu = 0.5, we got an f1 score of 0.86.
#with higher nu, the model captures more non linearities in the data, though running this needs
#more ram as nu = 0.5 means the model allows atleast half of the data points to be support vectors


#xgboost is another model which is designed to be sensitive towards the imbalanced data
#so we will be trying the xgboost model.
#ROC is effected by imbalanced data, so we prefer going with F1 score, but for the sake of
#looking at the metric we will compute the AUC_ROC for this model too.

#we will model 6 xgboosts one for each class, using tuning and having different hyperparameters
#seemes to be conflicting, so we are using the same predecided paremeters for all the models

#toxic
set.seed(1)
dataset2 = dataset
dataset2$toxic = as.factor(train$toxic)
levels(dataset2$toxic) = make.names(unique(dataset2$toxic))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$toxic, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = toxic ~ .
#kept method none and no grid options to speed up running things for the evaluator
#as the cv or boot strap or providing grid options take a large amount of time (more than 12 hours)
#and cannot be verified by the evaluator.
#We also have option to allow parallel processing in fitControl
fitControl <- trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid <- expand.grid(nrounds = 500,
                       max_depth = 6,
                       eta = .05,#learning rage
                       gamma = 0.001, #regularization parameter. 
                       colsample_bytree = .8,
                       min_child_weight = 1,
                       subsample = 1)


set.seed(13)

model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsToxic = predict(model,datasetTest,type = 'prob')$X1
predToxic = predict(model,datasetTest)

#severe_toxic calculation
set.seed(1)
dataset2 = dataset
dataset2$severe_toxic = train$severe_toxic
dataset2$severe_toxic = as.factor(dataset2$severe_toxic)
levels(dataset2$severe_toxic) = make.names(unique(dataset2$severe_toxic))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$severe_toxic, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = severe_toxic ~ .

set.seed(13)

model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsSevereToxic = predict(model,datasetTest,type = 'prob')$X1
predSevereToxic = predict(model,datasetTest)

#obscene
set.seed(1)
dataset2 = dataset
dataset2$obscene = train$obscene
dataset2$obscene = as.factor(dataset2$obscene)
levels(dataset2$obscene) = make.names(unique(dataset2$obscene))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$obscene, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = obscene ~ .
set.seed(13)
model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsObscene = predict(model,datasetTest,type = 'prob')$X1
predObscene = predict(model,datasetTest)

#threat
set.seed(1)
dataset2 = dataset
dataset2$threat = train$threat
dataset2$threat = as.factor(dataset2$threat)
levels(dataset2$threat) = make.names(unique(dataset2$threat))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$threat, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = threat ~ .
set.seed(13)
model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsThreat = predict(model,datasetTest,type = 'prob')$X1
predThreat = predict(model,datasetTest)

#insult
set.seed(1)
dataset2 = dataset
dataset2$insult = train$insult
dataset2$insult = as.factor(dataset2$insult)
levels(dataset2$insult) = make.names(unique(dataset2$insult))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$insult, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = insult ~ .
set.seed(13)
model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsInsult = predict(model,datasetTest,type = 'prob')$X1
predInsult = predict(model,datasetTest)

#identity_hate
set.seed(1)
dataset2 = dataset
dataset2$identity_hate = train$identity_hate
dataset2$identity_hate = as.factor(dataset2$identity_hate)
levels(dataset2$identity_hate) = make.names(unique(dataset2$identity_hate))
#uncomment below 2 lines for running on small data
#p <- createDataPartition(train$identity_hate, times = 1, p = 0.01) 
#dataset2 = dataset2[p[[1]], ]

formula = identity_hate ~ .
set.seed(13)
model = train(formula, data = dataset2,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)
print(summary(model))

predictionsHate = predict(model,datasetTest,type = 'prob')$X1
predHate = predict(model,datasetTest)

#performance evaluation
#AUC
submission <- sample_submission
submission$toxic = predictionsToxic
submission$severe_toxic = predictionsSevereToxic
submission$obscene = predictionsObscene
submission$threat = predictionsThreat
submission$insult = predictionsInsult
submission$identity_hate = predictionsHate

y_prb <- submission[(test_labels$toxic != -1 & test_labels$severe_toxic != -1 & test_labels$obscene != -1
                     & test_labels$threat != -1 & test_labels$insult != -1 & test_labels$identity_hate != -1), ]

toxic <- auc(y_act$toxic, y_prb$toxic)
severe_toxic <- auc(y_act$severe_toxic, y_prb$severe_toxic)
obscene <- auc(y_act$obscene, y_prb$obscene)
threat <- auc(y_act$threat, y_prb$threat)
insult <- auc(y_act$insult, y_prb$insult)
identity_hate <- auc(y_act$identity_hate, y_prb$identity_hate)
#macroaverage of auc
xgboostAUC <- (toxic + severe_toxic + obscene + threat + insult + identity_hate)/6
cat("AUC for xgboost", xgboostAUC, "\n")

#performance on xgboost predicting labels
#f1score
submission <- sample_submission
submission$toxic = factor(ifelse(predToxic=='X1', 1, 0), levels = c(0, 1))
submission$severe_toxic = factor(ifelse(predSevereToxic=='X1', 1, 0), levels = c(0, 1))
submission$obscene = factor(ifelse(predObscene=='X1', 1, 0), levels = c(0, 1))
submission$threat = factor(ifelse(predThreat=='X1', 1, 0), levels = c(0, 1))
submission$insult = factor(ifelse(predInsult=='X1', 1, 0), levels = c(0, 1))
submission$identity_hate = factor(ifelse(predHate=='X1', 1, 0), levels = c(0, 1))

xgb_lbl <- submission[(test_labels$toxic != -1 & test_labels$severe_toxic != -1 & test_labels$obscene != -1
                       & test_labels$threat != -1 & test_labels$insult != -1 & test_labels$identity_hate != -1), ]

if(length(unique(xgb_lbl$toxic)) == 1) {
  xgb_lbl$toxic[1] = (1 - xgb_lbl$toxic[1])
}
if(length(unique(xgb_lbl$severe_toxic)) == 1) {
  xgb_lbl$severe_toxic[1] = (1 - xgb_lbl$severe_toxic[1])
}
if(length(unique(xgb_lbl$threat)) == 1) {
  xgb_lbl$threat[1] = (1 - xgb_lbl$threat[1])
}
if(length(unique(xgb_lbl$obscene)) == 1) {
  xgb_lbl$Obscene[1] = (1 - xgb_lbl$obscene[1])
}
if(length(unique(xgb_lbl$insult)) == 1) {
  xgb_lbl$insult[1] = (1 - xgb_lbl$insult[1])
}
if(length(unique(xgb_lbl$identity_hate)) == 1) {
  xgb_lbl$identity_hate[1] = (1 - xgb_lbl$identity_hate[1])
}

toxic <- F1_Score(y_act$toxic, xgb_lbl$toxic)
severe_toxic <- F1_Score(y_act$severe_toxic, xgb_lbl$severe_toxic)
obscene <- F1_Score(y_act$obscene, xgb_lbl$obscene)
threat <- F1_Score(y_act$threat, xgb_lbl$threat)
insult <- F1_Score(y_act$insult, xgb_lbl$insult)
identity_hate <- F1_Score(y_act$identity_hate, xgb_lbl$identity_hate)
#macroaverage of f1
xgboostF1 <- (toxic + severe_toxic + obscene + threat + insult + identity_hate)/6
cat("f1 for xgboost", xgboostF1, "\n")

#Results
cat('one class svm at nu = 0.1 :', onesvmf1, "\n")
cat('xgboost:', xgboostF1, "\n")
cat('xgboost predicting probabililies', xgboostAUC, "\n")
#for xgboost we got very good scores of f1 and auc
#The high values of F1_Score and auc for both svm at nu = 0.5 and xgboost is due to the fact that the dataset
#being really controlled with out so much of a variety.

#in real world applications its rare to find such scores.



