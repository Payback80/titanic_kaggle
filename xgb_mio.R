library(data.table)
library(dplyr)
library(xgboost)
library(caret)
library(stringr)
library(rpart)

train <- fread("train.csv")
test  <- fread("test.csv")

combine <- bind_rows(train, test)
#check for NA , feature engineering 

sapply(combine, function(x) sum(is.na(x)))

combine <- mutate(combine,
                
                
               
          
                  FamilySize = (SibSp + Parch) + 1,
                  deck = if_else(is.na(Cabin),"U",str_sub(Cabin,1,1)),
                  family = SibSp + Parch,
                  alone = (SibSp == 0) & (Parch == 0),
                  large_family = FamilySize >= 4,
                  small_family = FamilySize <= 3
                  
)

combine$large_family <- as.numeric(combine$large_family)
combine$small_family <- as.numeric(combine$small_family)
combine$alone <- as.numeric(combine$alone)


# The number of titles are reduced to reduce the noise in the data
combine$Title <- sapply(combine$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combine$Title <- sub(' ', '', combine$Title)
#table(combi$Title)
combine$Title[combine$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combine$Title[combine$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combine$Title[combine$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combine$Title <- factor(combine$Title)  #remember to convert back to chr
#creating mother feature


#conversion for decision tree
combine$Sex <- as.factor(combine$Sex)
combine$Sex <- as.numeric(combine$Sex)-1
combine$deck <- as.factor(combine$deck)
combine$deck <- as.numeric(combine$deck)-1

# Decision trees model to fill in the missing Age values
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize + family + alone + large_family +small_family + deck + Title   , data=combine[!is.na(combine$Age),], method="anova")
combine$Age[is.na(combine$Age)] <- predict(Agefit, combine[is.na(combine$Age),])

# Fill in the Embarked and Fare missing values
#which(combi$Embarked == '')
combine$Embarked[c(62,830)] = "S"
combine$Embarked <- factor(combine$Embarked)
#which(is.na(combi$Fare))
combine$Fare[1044] <- median(combine$Fare, na.rm=TRUE)

combine <- mutate(combine,
                  young = factor( if_else(Age<=30, 1, 0, missing = 0) | (Title %in% c('Master.','Miss.','Mlle.')) ),
                  child = (Age<10),
                  master = Title == "Master"
)
combine$child <- as.numeric(combine$child)
combine$master <- as.numeric(combine$master)
               
                 


#select feature we want to use 
combine2 <- combine %>% select(Survived,
                               Pclass,
                               Sex,
                               Age,
                               SibSp,
                               Parch,
                               Fare,
                               Embarked,
                               family,
                               alone,
                               large_family,
                               small_family,
                               deck,
                               Title,
                               FamilySize,
                               young,
                               child,
                               master
                               )


#converting factors to numeric
# combine2$Sex <- as.factor(combine2$Sex)
# combine2$deck <- as.factor(combine2$deck)

combine2$Pclass <- as.numeric(combine2$Pclass)-1
# combine2$Sex <- as.numeric(combine2$Sex)-1
combine2$Embarked <- as.numeric(combine2$Embarked)-1
# combine2$deck <- as.numeric(combine2$deck) -1
combine2$Title <- as.numeric(combine2$Title)-1
combine2$young <- as.numeric(combine2$young)-1

#split again
train_2 <- combine2[1:891,]
test_2  <- combine2[892:1309,]
#generate train label 

train.label <- train_2$Survived
test.label <- test_2$Survived
#convert dataset to matrix
train_2<- as.matrix(train_2[,2:18])
test_2<- as.matrix(test_2[,2:18])

dtrain <- xgb.DMatrix(data = train_2, label = train.label)
dtest  <- xgb.DMatrix(data = test_2, label = test.label)

# View the number of rows and features of each set
dim(dtrain)
dim(dtest)

set.seed(1234)
# Set our hyperparameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              max_depth   = 8, #10
              subsample   = 1,
              eta         = 0.1,
              gammma      = 0,  #1
              colsample_bytree = 0.6,
              min_child_weight = 4)



cvFolds <- createFolds(combine2$Survived[!is.na(combine2$Survived)], k=10, list=TRUE, returnTrain=FALSE)

xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1000,
                 maximize = TRUE,
                 prediction = TRUE,
                 folds = cvFolds,
                 print_every_n = 10,
                 early_stopping_round = 100)


best_iter<- xgb_cv$best_iteration

# Pass in our hyperparameteres and train the model 
system.time(xgb <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = 112,
                           print_every_n = 100,
                           verbose = 1))

####################################################################



#################################################





##################################################
# Get the feature real names
names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model=xgb)[0:20] # View top 20 most important features

# Plot
xgb.plot.importance(importance_matrix)


# Prediction on test and train sets
pred_xgboost_test <- predict(xgb, dtest)
pred_xgboost_train <- predict(xgb, dtrain)

# Since xgboost gives a survival probability prediction, we need to find the best cut-off:
proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_xgboost_train<step,0,1)!=train.label)))
dim(proportion)

# Applying the best cut-off on the train set prediction for score checking
predict_xgboost_train <- ifelse(pred_xgboost_train<proportion[,which.min(proportion[2,])][1],0,1)
head(predict_xgboost_train)
score <- sum(train.label == predict_xgboost_train)/nrow(train)
score

# Applying the best cut-off on the test set
predict_xgboost_test <- ifelse(pred_xgboost_test<proportion[,which.min(proportion[2,])][1],0,1)

# Creating the submitting file
submit <- data.frame(PassengerId = combine[892:1309,c("PassengerId")], Survived = predict_xgboost_test)
write.csv(submit, file = "my_first_xgb3.csv", row.names = FALSE)



