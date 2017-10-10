#Part 3 of 3 for the exercise for the course Advanced Statistical Analysis and Machine Learning
#Joos Korstanje
#DSTI student (autumn 2016 cohort)
#DSTI email: joos.korstanje@edu.dsti.institute

##################### Description ###############
#This is my implementation of the vsurf package, following the article.

#I am aware of the warning messages, but they are not important for my implementation of the algorithm.

#Results:
#I have made one function that follows the description of page 4 of the vsurf article on moodle.
#my output is in a different format than in the article, but I tried to follow the mathematical approach as precisely as possible.

#At the end of the code there are 2 small examples to show that it works.

################################################

library(randomForest)
library(rpart)

################################################


select_variables<-function(dataset, formula, nRandomForest, dependent, objective){
  
  ##########################
  #Step 0: preparation work#
  ##########################
  
  #algorithm step 0: do nRanfomForest randomForests get the variable importances into a data frame
  #this has been split into the first tree and the other trees for convinience.
  
  #first tree
  #step 0: do a random forest
  data.rf=randomForest(formula=formula, data=dataset, importance=TRUE)
  #get the variable importance
  rf.importance_final=as.data.frame(importance(data.rf, 2))
  #add rownames to do a merge
  rf.importance_final$row_names=rownames(rf.importance_final)
  
  #for the other trees
  for(i in 2:nRandomForest){
    #step 0: do a random forest
    data.rf=randomForest(formula=formula, data=dataset, importance=TRUE)
    #get the variable importance
    rf.importance_df=as.data.frame(importance(data.rf, 2))
    #add rownames to do a merge
    rf.importance_df$row_names=rownames(rf.importance_df)
    rf.importance_final=merge(rf.importance_final, rf.importance_df, by="row_names")
  }
  
  #############################
  #Step 1 of vsurf publication#
  #############################
  
  #step1a take average and stdev of the values per variable
  means=apply(rf.importance_final[,2:ncol(rf.importance_final)], 1, mean)
  stdevs=apply(rf.importance_final[,2:ncol(rf.importance_final)], 1, sd)
  variables=rf.importance_final[,1]
  
  concat_step1a=data.frame(variables=variables,means=means, stdevs=stdevs)
  
  #step 1b: sort the variables by average importance
  concat_step1_sort=concat_step1a[order(-means),]
  #make sure to have the correct indexes (1 to nrow)
  rownames(concat_step1_sort)=seq(length=nrow(concat_step1_sort))
  
  #step 1c
  #run cart on y=stdevs, x=ranks of prediction
  concat_step1_sort$ranks=as.integer(rownames(concat_step1_sort))
  
  #find the cart tree that will lead to the threshold value to delete variables
  treshold_cart=rpart(concat_step1_sort$stdevs~concat_step1_sort$ranks)
  
  #get the actual trehsold value
  treshold_value=min(predict(treshold_cart, concat_step1_sort))
  #delete variables that have VI below this threshold value
  concat_step1c=concat_step1_sort[which(concat_step1_sort$means>treshold_value),]
  
  #############################
  #step 2 interpretation part#
  #############################
  #this part is fone in case of interpretation and in case of prediction
  
  rf_list_intermediate=vector()
  rf_list=vector()
  sd_list=vector()
  
  for(j in 1:nrow(concat_step1c)){
    #organizing the data to be able to go iteratively into randomForest
    #x data is the oe that is not given as being the dependent
    x = dataset[,which(colnames(dataset) != dependent)]
    #y is the one that is given in entry parameters as the dependent
    y = dataset[,dependent]
    
    #we need to enter the variables into the model in order of importance
    #this importance we get from concat step 1c
    x_order=concat_step1c$variables
    #then in every iteration we add one x variable to the model
    x_order_here=x_order[c(1:j)]
    
    #need to get everything in a dataframe to make the formula in the randomForest fucntion work
    y_df=data.frame(y=y)
    x_df=as.data.frame(x)
    #need to subset to select the correct number of variables in the dataframe
    x_df=x[,which(colnames(x) %in% x_order_here)]
    #make it into one overall dataframe
    rf_df=cbind.data.frame(y_df,x_df )
    
    #Run 25 times the RF algorithm:
    
    #for 25 forests (with this particular model), get the oob error and save it in a list
    for(i in 1:25){
      #step 0: do a random forest
      step2.rf=randomForest(y~.,data=rf_df, importance=TRUE, ntree=2000)
      
      #get out of bag error for classification (can get it directly)
      if(is.factor(y)){
        rf_list_intermediate[i]=mean(step2.rf$err.rate[,1])
      }
      #calculate oob error in case of regression (using formula)
      if(!is.factor(y)){
        rf_list_intermediate[i]=mean((step2.rf$predicted-y)^2)
      }
    }
    
    #rf_list is a vector with 25 mean oob errors from the 25 forests created per model
    #sd_list is a vector wih 25 sd values for the 25 forests per model
    #we need to calculate the average of the oob errors
    
    #save an average oob error per model
    rf_list[j] = mean(rf_list_intermediate)
    #get sd of oob errors
    sd_list[j] = sd(rf_list_intermediate)
    
    }
  
  #do the variabe selection:
  #first select models that have a oob < treshold
  #treshold is minimum oob + sd of minimum oob
  
  #get the index of the variable which has the minimum error. 
  min_error_index=which.min(rf_list)
  #get its oob error and sd
  min_error=rf_list[min_error_index]
  min_error_sd=sd_list[min_error_index]
  
  #calculate treshold
  oob_treshold=min_error+min_error_sd
  
  #get all models that have an oob error below the treshold
  suitable_index=which(rf_list<oob_treshold)
  
  #we want the smallest model that follows this treshold
  selected_model_index=min(suitable_index)
  
  #The selected variables are those that were included in the model with the minimum error
  selected_variables = concat_step1c$variables[1:selected_model_index]
  
  if(objective=="interpretation"){
    return(droplevels(selected_variables))
  }
  
  
  #######################
  #step 2 for prediction#
  #######################
  
  if(objective=="prediction"){
    #need to do some extra steps
    
    #stepwise selection of the variables selected in interpretation
    #backward, so starting with the most complete model
    #selected_variables is the selected variables from interpretation
    
    #will make a nested sequence
    #rf_list are the oob errors
    #rf_list differences are oob error differences
    increases=vector()
    #difference between current and next
    #if difference k is <= treshold, we will keep variables 1 to k (including k)
    for(k in 1:length(rf_list)-1){
      rf_list_increases = rf_list[k+1]-rf_list[k]
      increases[k]=rf_list_increases
    }
    
    decreases=0-increases
    
    #treshold: for notation see vsurf article
    m=length(rf_list)
    m_apostrophe=length(selected_variables)
    
    cst=1/(m-m_apostrophe)

    accumulator=vector()
    i=1
    for(l in m_apostrophe:m-1){
      errOOB_jplusone=rf_list[l+1]
      errOOB_j=rf_list[l]
      value=abs(errOOB_jplusone-errOOB_j)
      accumulator[i]=value
      i=i+1
    }
    
    #final tresholdvalue
    final_treshold_value=cst*sum(accumulator)
    
    #compare differences to treshold. 
    #difference>treshold is good
    #for the first value we find that is < treshold, we will keep only upto and including the 
    #index of the one that has this value < treshold (because of specficiaton of differences)
    
    #test if the indexes are sequential and if they are not, then stop
    if(decreases[1]<final_treshold_value){
      #take only one variable (the first), because there is not enough decrease
      selected_variables=selected_variables[1]
    }
    else{
      #select variables by treshold of decrease and
      #get the variable names of these variabes into the output vector
      selected_variables_predict=vector()
      #we certainly need the first variable because we are in the else
      selected_variables_predict[1]=selected_variables[1]
      m=1
      while(decreases[m]>final_treshold_value){
        #need to take the m+1 because the decrease vector is the decrease between m and m+1
        #so if decrease m is large enough, we want to have varaible m+1 included
        selected_variables_predict[m+1]=selected_variables[m+1]
        m=m+1
        }
    }
    return(droplevels(selected_variables))
  }
}

########################################################
################Examples################################
########################################################

#Please note that examples are not meant to show a model with a useful interpretation.
#They just show that the functions work and how to use them.

#########################################
#iris example - classification framework#
#########################################
select_variables(dataset=iris, formula=as.formula(Species ~ .), 50, dependent="Species", objective="interpretation")
select_variables(dataset=iris, formula=as.formula(Species ~ .), 50, dependent="Species", objective="prediction")

###############################################
#car.test.frame example - regression framework#
###############################################
select_variables(dataset=car.test.frame[complete.cases(car.test.frame),], formula=as.formula(Price ~ Country + Reliability + Mileage + Type + Weight + Disp. + HP), 50, dependent="Price", objective="interpretation")
select_variables(dataset=car.test.frame[complete.cases(car.test.frame),], formula=as.formula(Price ~ Country + Reliability + Mileage + Type + Weight + Disp. + HP), 50, dependent="Price", objective="prediction")
