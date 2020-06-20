if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

#original dataset available from - https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420?select=pokedex_%28Update_05.20%29.csv
#due to need to need for kaggle login dataset is provided in git


#read csv into dataframe
fulldataset <- read.csv(file = "datasets_593561_1119602_pokedex_(Update_05.20).csv")

#check numbers of rows and columns
ncol(fulldataset)
nrow(fulldataset)

# check the dataset names and classes
str(fulldataset)

#remove unneeded columns X - a holdover column number, german and japanese names
fulldataset$X <- NULL
fulldataset$german_name <- NULL
fulldataset$japanese_name <- NULL

#our goal is to predict the status of a pokemon given. The 'against_x' 'columns refer to in game damage multiplication 
# values and are drawn from the types field(s), as such they will not be used by us and can be cut
names(fulldataset)
fulldataset[31:48] <- NULL

#check the set again
names(fulldataset)

#using https://bulbapedia.bulbagarden.net/wiki/Main_Page, abilities are able to be changed by the trainer. Hidden abilities
#are also either dependent upon the personality or the unique to the pokemon dependent upon which information you read
#as such the changability of these means they will not translate well to a general predictive model and will be removed.
fulldataset$abilities_number <- NULL
fulldataset$ability_1 <- NULL
fulldataset$ability_2 <- NULL
fulldataset$ability_hidden <- NULL

#looking at the species column can see some anomolies
fulldataset$species[1:10]

#it appears the accented e in pokemon is appearing as incorrect characters. looking further at this over half of the dataset
#has a unique entry in this field
fulldataset %>% group_by(species) %>%
  summarise(no=n()) %>%
  group_by(no) %>%
  summarise(n())

#and amongst the less frequent of our classification targets
fulldataset %>% 
  filter(status%in%c("Legendary", "Mythical", "Sub Legendary")) %>%
  group_by(status,species) %>%
  summarise(no=n()) %>%
  group_by(no) %>%
  summarise(n())

#similar ratio, given this is not anticipated to be a valueable prediction target, and given the prevalence
#of uniqe entries will not translate to new datasources. As such we will remove this column and not use in our analysis.
fulldataset$species <- NULL

#reading about the other egg values, information is inconsistent and displayed in any game, only "mentioned canonically".
#As such the data in these field is deemed to be unreliable and will not be used. 
fulldataset$egg_type_1 <- NULL
fulldataset$egg_type_2 <- NULL
fulldataset$egg_type_number <- NULL

#now will check for NAs - here is a function to do that
check_na <- function(x){
  any(is.na(x))
}

#running the NA function
check_na(fulldataset)

#there are some so will look at summary as initial look at where they are
summary(fulldataset)

#we'll fix NAs. First weight, with only one NA
fulldataset %>% filter(is.na(weight_kg))

#to fix this well reference external data https://bulbapedia.bulbagarden.net/wiki/Eternatus_(Pok%C3%A9mon) and increase
#weight from previous evolution by the same factor as height
fulldataset$weight_kg <- replace_na(fulldataset$weight_kg, 950*5)

#catch rate, base_friendship and Base_experience all have 104 missing entries. Different sites offer different values and
#explanations for these. With no relaible source, and no way to know how these may affect our modelling if we choose either
#minimum , mean, or out of scope values for example, we will remove these features. 
#The same will apply to the percentage male with 236 NAs.
fulldataset[17:19] <- NULL
fulldataset$percentage_male <- NULL

# finally we will look at egg cycles, here there is one entry. we will look at that.
fulldataset %>% filter(is.na(egg_cycles)) 
#it appears that this pokemon is part of a set. We will therefore update its egg and other values to match those of the set
which(fulldataset$pokedex_number==555)
fulldataset[653,]$egg_cycles <- 20
#also can see that growth rate is blank in this entry (from summary and here), so will update this too
fulldataset[653,]$growth_rate <- "Medium Slow"
#re-factor the growth rate column
fulldataset$growth_rate <- factor(fulldataset$growth_rate)
str(fulldataset)
#no we have changed that, lets look for NAs again
check_na(fulldataset)
#Now we have a fulldataset ready to go.

#note name and pokedex number left in as they will be useful descriptors for people to see, but won't be used for the
#modelling
#prepared dataset has 
nrow(fulldataset)
ncol(fulldataset)

#split into test and train datasets. Test will only be used for final results. Due to relatively small dataset
#will not split into three sets - test, train and validation - and will use cross validation on the training set
#to tune models

#will split the data using createdatapartion function to ensure spread of the target classifications
#and use a 9:1 training to test ratio as we want as much data to work with to train the models with only a small 
#total dataset and still enough to prove a valueable final test

#set the seed to enable repeatability
set.seed(1, sample.kind = "Rounding")
#split dataset into 9:1 chunk to form working data and validation data
test_index <- createDataPartition(fulldataset$status, times = 1, p = 0.1, list = FALSE)
testset <- fulldataset[test_index,]
trainset <- fulldataset[-test_index,]


#-------------------------------EDA
#note name and pokedex number left in as they will be useful descriptors for people to see, but won't be used for the
#modelling
#prepared dataset has 
nrow(trainset)
ncol(trainset)
nrow(testset)
ncol(testset)


#quick preview (split for easier visibility)
head(trainset[1:6])
head(trainset[7:12])
head(trainset[13:18])


#we are looking to classify the status of a pokemon
trainset %>% 
  group_by(status) %>% 
  summarise(`number of pokemon`=n())
#the features available to do this can be split into two rough categories that we will focus on -
#descriptive [1:9], ability [10:21] 

#being looking at ability data
trainset %>% 
  group_by(status) %>% 
  summarise(`Mean total points`=mean(total_points),
            `Mean hp`=mean(hp),
            `Mean attack`=mean(attack),
            `Mean defense`=mean(defense),
            `Mean sp_attack`=mean(sp_attack),
            `Mean sp_defense`=mean(sp_defense),
            `Mean speed`=mean(speed))
# from this we can see that the mean for legendary pokemon have, with the expection of speed, higher mean values than
#the others, and Normal pokemon have the lowest, with the mythical and sublegendary share the second and third positions
#for the others.

#lets visualise some of this
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

#these show a similar story, however highlight that some outliers, particularly in the legendary and normal 
#status groups are far from the mean and may influence the models

#using the generations factors we will see if these have been consistent over time.
#split into status by generation and summarise the mean of the total points

trainset %>%
  group_by(status, generation) %>%
  summarise(mean_total_points = mean(total_points)) %>%
  ggplot(aes(generation, mean_total_points, color=status)) +
  geom_line()

#can see that normal status is fairly consistent, the others dip in gen 7, and the mean total is distinct for each generation
#this the could be a useful feature along with the stats to selec which status more effectively. 

#looking at the number of   
trainset %>%
  group_by(status,type_number) %>%
  summarise(n())
#number of types seems fairly consistently split across status, how about the specifics

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

#from these we can see there are no standout unique values, however several only appear in two status groupings and be a
#useful model feature

#given the specifics data we have, we can remove the type_number column from our set

trainset$type_number <- NULL
testset$type_number <- NULL

#Egg cycles are the amount of times that a specific number of steps must be taken in game
#before a pokemon hatches https://bulbapedia.bulbagarden.net/wiki/Egg_cycle. looking at the graph:
trainset %>% 
  group_by(status, egg_cycles) %>%
  summarise(count=n()) %>% 
  ggplot(aes(egg_cycles,count, color=status )) +
  geom_point()

# we see that higher egg cycles are reserved for non normal pokemon

trainset %>% 
  group_by(status, egg_cycles) %>%
  summarise(count=n())

#looking at height and weight:

trainset %>% 
  ggplot(aes(height_m, weight_kg, color=status)) + 
  geom_point() + 
  scale_x_log10() + 
  scale_y_log10()

#mythical are relatively spread across the range, however normal and legendary/sublegendary have a visible grouping with
#only few outliers

# Finally we will look indepentently at the growth before combining values and looking for correlations etc

trainset %>% 
  group_by(status, growth_rate) %>% 
  summarise(count=n()) %>% 
  ggplot(aes(growth_rate, count, color=status)) +
  geom_point()

#here we see a further grouping as all non-normal pokemon are either medium-slow or slow.


#using combinations of growth_rate, egg_cycles, total points and height*weight(which already showed grouping), we can 
#see other groupings that show these may be god features for a predictive model

trainset %>% ggplot(aes(growth_rate, egg_cycles, color=status)) + geom_point()

trainset %>% ggplot(aes(growth_rate, total_points, color=status)) + geom_point()

trainset %>% ggplot(aes(egg_cycles, total_points, color=status)) + geom_point()

trainset %>% ggplot(aes(height_m*weight_kg, total_points, color=status)) + geom_point() + scale_x_log10() + scale_y_log10()

trainset %>% ggplot(aes(height_m*weight_kg, egg_cycles, color=status)) + geom_point()+ scale_x_log10()

trainset %>% ggplot(aes(height_m*weight_kg, growth_rate, color=status)) + geom_point() + scale_x_log10()

#------------------------modelling

#this is a multiple classification task. We will explore three methods for modelling this, KNN, Random Forest(rf) and Support
# vector Machine(svm). The selection of these is informed by two factors 1) they are all capable of multiple classification,
#2) they are from different 'families' of algorithm. KNN was chosen to display the capabilities taught in the PH125.9x
#course and will be tuned by k, the number of neighbours considered, RF will be used for the same reason as it was shown to be the top performing
#general usecase algorithm in delgado(2014), second by Caruana(20XX), and best (in binary classifications it must be stated),
#by Wainer et. al (2016). The RF model will be tuneed using mtry - number of random features considered at each split - and
#ntree. The Choice of SVM was to push myself to learn about a different method to those on the course and because in the 
# aforementioned papers SVM were highly rated at completing this type of task. The SVM will be tried with a liner and radial
#kernal, and tuned using cost. 

#to enable knn and SVM we need to normalise the data and account for factors. This is because different scales can warp the
#algorithm. We will therefore perform one-hot encoding, to create unique variables of 1 or 0 for each factor and 
#normalise the data. These values from the training data will then be used to account for the factors on the test set, and
#the min/max of the training data used to normalise the numeric values. This will prevent inadvertant improvement
#of the training data from the test set byt accounting for factors and normalising individually or as a whole dataset.

#--preparing for modelling--#

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




##--KNN model--#
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

#increasing the number of trees has no effect upon our max results, so we will use the 500 tree model, just because


##--svm model--##

#set the seed to enable replication, same seed to provide fair test between algorithms
set.seed(2, sample.kind = "Rounding")
#using the same cross validation we'll first train a linear model using a cost tune of 0.001-1000 then refine
svm_l_model <- train(status~.,
                     method="svmLinear",
                     data=normtrainset,
                     trControl=control,
                     tuneGrid=data.frame(C=(0.0001 * 10^(seq(0,6,2)))))
svm_l_model
ggplot(svm_l_model)
svm_l_model$bestTune
max(svm_l_model$results$Accuracy)
# now will tune closer to this value to refine
set.seed(2, sample.kind = "Rounding")
svm_l_model <- train(status~., 
                     method="svmLinear", 
                     data=normtrainset, 
                     trControl=control, tuneGrid=data.frame(C=seq(0.001,0.2,length=30)))
svm_l_model
ggplot(svm_l_model)
svm_l_model$bestTune
svm_l_acc <- max(svm_l_model$results$Accuracy)


#repeat for the radial kernel

set.seed(2, sample.kind = "Rounding")
svm_r_model <- train(status~.,
                     method="svmRadialCost",
                     data=normtrainset,
                     trControl=control,
                     tuneGrid=data.frame(C=(0.0001 * 10^(seq(0,6,2)))))

svm_r_model
ggplot(svm_r_model)
svm_r_model$bestTune
max(svm_r_model$results$Accuracy)

set.seed(2, sample.kind = "Rounding")
svm_r_model <- train(status~., 
                     method="svmRadialCost", 
                     data=normtrainset, 
                     trControl=control, tuneGrid=data.frame(C=seq(0.1,2,length=30)))


svm_r_model
ggplot(svm_r_model)
svm_r_model$bestTune
svm_r_acc <- max(svm_r_model$results$Accuracy)

##----model comparison----##

#add the models and their accuracy into df for easy viewing
comparison <- data.frame(model=c("KNN", "Random Forest", "SVM Linear", "SVM Radial"), 
                         accuracy=c(knn_acc, rf_acc, svm_l_acc, svm_r_acc))

comparison



##----------results----------##

#since the Random forest model is our best performing, our final test will be to apply our model to the untouched
# test data
set.seed(3, sample.kind = "Rounding")
final_predictions <- predict(rf_model, normtestset)
#use a confusion matrix to view
confusionMatrix(data = final_predictions, reference = testset$status)
