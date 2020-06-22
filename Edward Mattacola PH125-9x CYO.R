#download packages if not present an load them
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

####---------Cleaning up the dataset

#original dataset available from - 
#https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420?select=pokedex_%28Update_05.20%29.csv
#read csv into dataframe
fulldataset <- read.csv(file = "datasets_593561_1119602_pokedex_(Update_05.20).csv")

#check numbers of rows and columns
ncol(fulldataset)
nrow(fulldataset)

#check the dataset names and classes
str(fulldataset)

#remove unneeded columns 
fulldataset$X <- NULL
fulldataset$german_name <- NULL
fulldataset$japanese_name <- NULL

#remove unndeed columns due to unneeded data
fulldataset[31:48] <- NULL
names(fulldataset)
fulldataset$abilities_number <- NULL
fulldataset$ability_1 <- NULL
fulldataset$ability_2 <- NULL
fulldataset$ability_hidden <- NULL

#looking at the species column
fulldataset$species[1:10]

#look deeper at species
fulldataset %>% group_by(species) %>%
  summarise(no=n()) %>%
  group_by(no) %>%
  summarise(n())

#Look specifically at less frequent status
fulldataset %>% 
  filter(status%in%c("Legendary", "Mythical", "Sub Legendary")) %>%
  group_by(status,species) %>%
  summarise(no=n()) %>%
  group_by(no) %>%
  summarise(n())

#remove unneeded columns due to unreliable data
fulldataset$species <- NULL
fulldataset$egg_type_1 <- NULL
fulldataset$egg_type_2 <- NULL
fulldataset$egg_type_number <- NULL

#function to check NA entries
check_na <- function(x){
  any(is.na(x))
}

#running the NA function
check_na(fulldataset)

#Look at summary for more NA detail
summary(fulldataset)

#look at weight NA category
fulldataset %>% filter(is.na(weight_kg))

#fix weight NA by multiplying previous evolution weigh by 
fulldataset$weight_kg <- replace_na(fulldataset$weight_kg, 950*5)

#remove unneeded columns due to multiple NAs and unreliable data
fulldataset[17:19] <- NULL
fulldataset$percentage_male <- NULL

#Look at NA in egg cycles
fulldataset %>% filter(is.na(egg_cycles)) 

#Pokemon is part of a set, will update egg values to match
which(fulldataset$pokedex_number==555)
fulldataset[653,]$egg_cycles <- 20

#same pokemon has blank growth rate, will update this to match to 
fulldataset[653,]$growth_rate <- "Medium Slow"

#re-factor the growth rate column
fulldataset$growth_rate <- factor(fulldataset$growth_rate)
str(fulldataset)

#no we have changed that, lets look for NAs again
check_na(fulldataset)

#Now have a clean dataset

####--------------Splitting into test and train sets.

#look at the columns and rows
nrow(fulldataset)
ncol(fulldataset)

#set the seed to enable repeatability
set.seed(1, sample.kind = "Rounding")
#split dataset into 9:1 chunk to form training data and test data
test_index <- createDataPartition(fulldataset$status, times = 1, p = 0.1, list = FALSE)
testset <- fulldataset[test_index,]
trainset <- fulldataset[-test_index,]


####-------------------------------EDA

#look at the numbers of columns and rows
nrow(trainset)
ncol(trainset)
nrow(testset)
ncol(testset)

#investigate head of training, split for visibility
head(trainset[1:6])
head(trainset[7:12])
head(trainset[13:18])

#Target variable is status, group by this variable and look at numbers and percentage
trainset %>% 
  group_by(status) %>% 
  summarise(`number of pokemon`=n(), percentage=n()/nrow(trainset)*100)

#Summarise the mean values of combat points grouped by status
trainset %>% 
  group_by(status) %>% 
  summarise(`Mean total points`=mean(total_points),
            `Mean hp`=mean(hp),
            `Mean attack`=mean(attack),
            `Mean defense`=mean(defense),
            `Mean sp_attack`=mean(sp_attack),
            `Mean sp_defense`=mean(sp_defense),
            `Mean speed`=mean(speed))

#Visualise the data in boxplots for easier analysis and to see outliers
trainset %>% ggplot(aes(status, total_points, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, hp, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, attack, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, defense, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, sp_attack, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, sp_defense, fill=status)) + 
  geom_boxplot()

trainset %>% ggplot(aes(status, speed, fill=status)) + 
  geom_boxplot()


#split into status by generation and summarise the mean of the total points to see changes over gerneration
trainset %>%
  group_by(status, generation) %>%
  summarise(mean_total_points = mean(total_points)) %>%
  ggplot(aes(generation, mean_total_points, color=status)) +
  geom_line()

#looking at the number of types
trainset %>%
  group_by(status,type_number) %>%
  summarise(n())

#number of types seems fairly consistently split across status, look at the specifics
trainset %>%
  group_by(status, type_1) %>%
  summarise(count = n()) %>%
  ggplot(aes(type_1, count, fill=status)) +
  geom_bar(stat = "Identity")

trainset %>%
  group_by(status, type_2) %>%
  summarise(count=n()) %>%
  ggplot(aes(type_2,count, fill=status)) +
  geom_bar(stat="Identity")

#given the specifics data remove the type number as its unneeded
trainset$type_number <- NULL
testset$type_number <- NULL

#look at egg cycles
trainset %>% 
  group_by(status, egg_cycles) %>%
  summarise(count=n()) %>% 
  ggplot(aes(egg_cycles,count, color=status )) +
  geom_point()

#see saem data in a table, higher values for non normal pokemon
trainset %>% 
  group_by(status, egg_cycles) %>%
  summarise(count=n())

#looking at height and weight
trainset %>% 
  ggplot(aes(height_m, weight_kg, color=status)) + 
  geom_point() + 
  scale_x_log10() + 
  scale_y_log10()
#some apparent groupings


#look at growth rate
trainset %>% 
  group_by(status, growth_rate) %>% 
  summarise(count=n()) %>% 
  ggplot(aes(growth_rate, count, color=status)) +
  geom_point()
# similar grouping and split as egg cycles

#graph the vaiables looked at independently against one another to look for patterns
trainset %>% 
  ggplot(aes(growth_rate, egg_cycles, color=status)) +
  geom_point()

trainset %>% 
  ggplot(aes(growth_rate, total_points, color=status)) + 
  geom_point()

trainset %>% 
  ggplot(aes(egg_cycles, total_points, color=status)) + 
  geom_point()

trainset %>% 
  ggplot(aes(height_m*weight_kg, total_points, color=status)) + 
  geom_point() + 
  scale_x_log10() + 
  scale_y_log10()

trainset %>% 
  ggplot(aes(height_m*weight_kg, egg_cycles, color=status)) + 
  geom_point() + 
  scale_x_log10()

trainset %>% 
  ggplot(aes(height_m*weight_kg, growth_rate, color=status)) + 
  geom_point() + 
  scale_x_log10()

######------------------------modelling

#--preparing for modelling--#

#remove unneeded columns
trainset$pokedex_number <- NULL
testset$pokedex_number <- NULL
trainset$name <- NULL
testset$name <- NULL

#Creating dummy variables by converting factors to as many binary variables as here are categories.
dummies_model <- dummyVars(status ~ ., data=trainset)
#using the dummy categories make a training dataset replacing the factor values with the new binary value columns
normtrainset <- predict(dummies_model, newdata = trainset)
#turn that into a dataframe for ease of use
normtrainset <- data.frame(normtrainset)

#apply the same to the test set
normtestset <- predict(dummies_model, newdata = testset)
normtestset <- data.frame(normtestset)

#create a normalisation model using the training data - method range is normalising between 0-1
normalise_model <- preProcess(normtrainset, method='range')
#apply model to the normtrainset (the one with factors accounted for)
normtrainset <- predict(normalise_model, newdata = normtrainset)
#turn to dataframe
normtrainset <- data.frame(normtrainset)

# apply the same to the test set
normtestset <- predict(normalise_model, newdata = normtestset)
normtestset <- data.frame(normtestset)

#add our status column back to the datasets
normtrainset<- normtrainset %>% mutate(status=trainset$status)
normtestset <- normtestset %>% mutate(status=testset$status)




##----KNN model-----##

#set the seed to enable replication
set.seed(2, sample.kind = "Rounding")
#select our training control options -  10 fold cross validation repeated 3 times
control <- trainControl(method="repeatedcv",number=10, repeats = 3)
#create our model with caret package, traincontrol options as before, tunlength is 20 k vairables starting from 5
# in increments of 2
knn_model <- train(status~., method="knn", data=normtrainset, trControl=control, tuneLength=20)
#plot of the model accuracy based upon k value 
ggplot(knn_model)
#model best output k value - 
knn_model$bestTune
knn_acc <- max(knn_model$results$Accuracy)


##--RF model--#

####WARNING this will take a while to run - avg 1 min per 100 trees of rf model on i7-8700T with 16GB RAM

#set the seed to enable replication, same seed to provide fair test between algorithms
set.seed(2, sample.kind = "Rounding")
#generate rf model with default ntrees of 500, same 10 fold cross validation repeated 3 times and 20 mtry values
#starting from 2 and increasing in intervals of 3
rf_model <- train(status~., method="rf", data=normtrainset, trControl=control, tuneLength=20)
#plot of mtry vs accuracy for rf model
ggplot(rf_model)
#which mtry value was best
rf_model$bestTune
#and what was the accuracy for comparison
rf_acc <- max(rf_model$results$Accuracy)
#look at the most important variables in the model
varImp(rf_model)

#perform the same but with ntree 1000
set.seed(2, sample.kind = "Rounding")
#generate rf model with default ntrees of 1000, same 10 fold cross validation repeated 3 times and 20 mtry values
#starting from 2 and increasing in intervals of 3
rf_model_1k <- train(status~., method="rf", data=normtrainset,ntree=1000, trControl=control, tuneLength=20)
#plot of mtry vs accuracy for rf model
ggplot(rf_model_1k)
#which mtry value was best
rf_model_1k$bestTune
#and what was the accuracy for comparison
rf_1k_acc <- max(rf_model_1k$results$Accuracy)
#look at the most important variables in the model
varImp(rf_model_1k)


#perform the same but with ntree 2000
set.seed(2, sample.kind = "Rounding")
#generate rf model with default ntrees of 2000, same 10 fold cross validation repeated 3 times and 20 mtry values
#starting from 2 and increasing in intervals of 3
rf_model_2k <- train(status~., method="rf", data=normtrainset,ntree=2000, trControl=control, tuneLength=20)
#plot of mtry vs accuracy for rf model
ggplot(rf_model_2k)
#which mtry value was best
rf_model_2k$bestTune
#and what was the accuracy for comparison
rf_2k_acc <- max(rf_model_2k$results$Accuracy)
#look at the most important variables in the model
varImp(rf_model_2k)

##--svm model--##

#set the seed to enable replication, same seed to provide fair test between algorithms
set.seed(2, sample.kind = "Rounding")
#using the same cross validation train a linear model using a cost tune of 0.001-1000 then refine
svm_l_model <- train(status~.,
                     method="svmLinear",
                     data=normtrainset,
                     trControl=control,
                     tuneGrid=data.frame(C=(0.0001 * 10^(seq(0,6,2)))))

# plot of cost vs accuracy
ggplot(svm_l_model)
#which best cost 
svm_l_model$bestTune
#what was the accuracy with first tune
max(svm_l_model$results$Accuracy)
# now will tune closer to this cost value to refine
# set seed to enable replication
set.seed(2, sample.kind = "Rounding")
#train svm linear model with same cv and 30 cost values from 0.001-0.2
svm_l_model <- train(status~., 
                     method="svmLinear", 
                     data=normtrainset, 
                     trControl=control, tuneGrid=data.frame(C=seq(0.001,0.2,length=30)))
# graph of cost vs accuracy
ggplot(svm_l_model)
#best cost values
svm_l_model$bestTune
#and what was the accuracy for comparison
svm_l_acc <- max(svm_l_model$results$Accuracy)


#repeat for the radial kernel

#set seed for replication
set.seed(2, sample.kind = "Rounding")
#train model using radial kernel and cost tune of 0.001-1000
svm_r_model <- train(status~.,
                     method="svmRadialCost",
                     data=normtrainset,
                     trControl=control,
                     tuneGrid=data.frame(C=(0.0001 * 10^(seq(0,6,2)))))
#plot of cost vs accuracy for radial model
ggplot(svm_r_model)
#best cost
svm_r_model$bestTune
#and accuracy
max(svm_r_model$results$Accuracy)

#refine with costs closer to initial optimum

#set seed for replication
set.seed(2, sample.kind = "Rounding")
#train model using 30 cost values from 0.1 to 0.2
svm_r_model <- train(status~., 
                     method="svmRadialCost", 
                     data=normtrainset, 
                     trControl=control, tuneGrid=data.frame(C=seq(0.1,2,length=30)))
#plot of cost vs accuracy
ggplot(svm_r_model)
#best cost value
svm_r_model$bestTune
#accuracy for comparison
svm_r_acc <- max(svm_r_model$results$Accuracy) 

##----model comparison----##

#add the models and their accuracy into df for easy viewing
comparison <- data.frame(model=c("KNN", "Random Forest", "Random Forest 1k", "Random Forest 2k", "SVM Linear", "SVM Radial"), 
                         accuracy=c(knn_acc, rf_acc, rf_1k_acc, rf_2k_acc, svm_l_acc, svm_r_acc))
#veiw comparison
comparison

##----------results----------##

#random forest 100 trees rf_model_1k was best, will use to predict against test set
#set seed for replication 
set.seed(3, sample.kind = "Rounding")
#use predict function to predict using our model and the noramlised test set
final_predictions <- predict(rf_1k_model, normtestset)
#use a confusion matrix to view
confusionMatrix(data = final_predictions, reference = testset$status)
