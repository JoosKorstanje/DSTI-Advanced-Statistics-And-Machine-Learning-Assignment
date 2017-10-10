#Part 1 of 3 for the exercise for the course Advanced Statistical Analysis and Machine Learning
#Joos Korstanje
#DSTI student (autumn 2016 cohort)
#DSTI email: joos.korstanje@edu.dsti.institute



##################### Description ###############

#Answer to question 1: "Produce a function to be able to perform the cross validation associated to the cart algorithm"

#Explanatory note: I have produced two functions that can be used (together) to do cross-validation of the cart algorithm. 
#I thereby used the rpart implementation of the cart algorithm that we have seen in class.

#There will be one cross-validation error calculated for one time that a user runs 
#this code (and therefore he runs one time the rpart CART algorithm). I have used the formula for
# cross validation error that we have seen in class.

#The result of this code could tell its user how to choose between different trees. For a tree given by the rpart algorithm (based on a model forumula),
# my code will give the cross validation error. The user can then decide which tree he prefers (based on which one has the minimum cv error)

#In this script I present two functions, and then two very small example applciations in which I show that the code is working.

#############################################################################

library(rpart)

#############################################################################

# Function to split a dataset into k folds
get_crossval_folds<-function(dataframe, n_folds){
  #initialize a list to store the different splits
  dataset_split = list()
  
  #compute the number of values per split (in total you need the same number of observations)
  fold_size = (nrow(dataframe) / n_folds)
  
  #loop through the k folds to be created
  for(i in 1:n_folds){
    
    #take a sample from the data: first get the index
    MySampleIndex=sample(x=rownames(dataframe), size=fold_size)
    #then do the subset
    MySample=dataframe[MySampleIndex,]
    
    #and store it in the list
    dataset_split[i]=list(MySample)
    
    #we want to avoid having replacement, so delete it from the original data
    dataframe=dataframe[!rownames(dataframe) %in% rownames(MySample),]
    
  }
  return(dataset_split)
}

################################################################################

# Evaluating rpart using specified cross validation folds
get_crossvalidation_error<-function(dataset, n_folds, y, f){
  
  folds = get_crossval_folds(dataset, n_folds)
  
  #initialize a list for the scores
  scores = list()
  cv_error_list = list()
  
  #for each of the folds (subsets of the data)
  for(i in 1:n_folds){
    
    #copy folds and call it train set
    train_set_list = folds
    
    #get the folds from the trainset into a dataframe
    train_set_df = do.call("rbind", train_set_list)
    
    #the left-out data subset will become the test set
    test_set_df = as.data.frame(train_set_list[i])
    
    #leave out one of the folds (leave one out cross validation) 
    train_set_list = train_set_list[-i]
    
    #Doing the comparison in several steps:
    #getting the model,
    trained_model = rpart(formula=f, data=train_set_df)
    
    if(!is.factor(dataset[,y])){
      #using it to make predictions on the test set
      test_predictions = predict(trained_model, test_set_df, type="vector")
    }
    
    if(is.factor(dataset[,y])){
      #using it to make predictions on the test set
      test_predictions = predict(trained_model, test_set_df, type="class")
    }
    
    #getting the actual, correct answers
    real_values = test_set_df[,y]
    
    #save the predictions
    scores[i] = list(test_predictions)
    
    #calculating the cv value, different method depending on whether classification or regresion framework
    
    #Regression framework, when y is not factor
    if(!is.factor(dataset[,y])){
      
      #compute cv error
      cv_error_scalar = sum((real_values-test_predictions)^2)
      
      #save the error
      cv_error_list[i]=cv_error_scalar
    }
    
    #Classification framework, when y is a factor
    if(is.factor(dataset[,y])){
      
      #create a vector that indicates whether the classification was good or wrong
      correct_vector=real_values==test_predictions
      
      #keep only the ones that were wrong
      wrong_vector=correct_vector[which(correct_vector==FALSE)]
      
      #count the number of wrong answers
      number_wrong_scalar=length(wrong_vector)
      
      #compute cv error as percentage of wrong answers
      fold_size = (nrow(dataset) / n_folds)
      cv_error_scalar = number_wrong_scalar / fold_size
      
      #save the cv error
      cv_error_list[i]=cv_error_scalar
    }
  }
  
  cv_sum=0
  
  for(i in 1:n_folds){
    cv_sum = cv_sum + cv_error_list[[i]]
  }
  
  fold_size = (nrow(dataset) / n_folds)
  
  #Concatenate the cv error into one value (the rpart algorithm executes only one tree and we therefore need only one cv error value)
  #formula for cv error (sum of errors * (fold size / n observations)
  cv_error = cv_sum * (fold_size / nrow(dataset))
  
  return(cv_error)
}


######################################################################################
##########Here I execute this function on the car.test.frame to show that it works####
###############in both the classification and regression framework if it works########
######################################################################################

#Please note that examples are not meant to show a model with a useful interpretation.
#They just show that the functions work and how to use them.

car.test.frame[,"Type"] = as.factor(car.test.frame[,"Type"])

###################################################
#Regression framework example using car.test.frame#
###################################################
f = Price ~ Country + Reliability + Mileage + Type + Weight + Disp. + HP
get_crossvalidation_error(car.test.frame, 5, "Price", f)

#######################################################
#Classification framework example using car.test.frame#
#######################################################
f2 = Type ~ Price + Country + Reliability + Mileage + Weight + Disp. + HP
get_crossvalidation_error(car.test.frame, 5, "Type", f2)

