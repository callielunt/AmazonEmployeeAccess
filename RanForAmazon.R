#### Forrest ####

# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ggplot2)
library(doParallel)
library(kknn)

#Parallel Computing
num_cores <- parallel::detectCores()
cl <- makePSOCKcluster(num_cores/2)
registerDoParallel(cl)

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

# Recipe
my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

# Model
## Define a model
my_mod_rf <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

## Create a workflow w/ model and recipe

#Workflow
randfor_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod_rf)
# 
#  prepped_recipe <- prep(bike_recipe)
#  final <- bake(prepped_recipe, new_data = new_train)


## Set up a grid of tuning values
grid_of_tuning_params_rf <- grid_regular(mtry(range = c(1, 10)),
                                         min_n(),
                                         levels = 5)

## Set up K-fold CV
folds_rf <- vfold_cv(train, v = 5, repeats = 1)

## Find best tuning parameters
CV_results_rf <- randfor_wf %>% 
  tune_grid(resamples = folds_rf,
            grid = grid_of_tuning_params_rf,
            metrics = metric_set(roc_auc))

bestTune_rf <- CV_results_rf %>% 
  select_best(metric = "roc_auc")

## Finalize workflow and predict 
final_wf_rf <-
  randfor_wf %>% 
  finalize_workflow(bestTune_rf) %>% 
  fit(data = train)

## Predict
# make predictions
predictions <- predict(final_wf_rf,
                              new_data = test,
                              type = "prob")

submission_knn <- predictions |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_knn, file = "./RanForPred.csv", delim = ",")


# Stop Parallel computing
stopCluster(cl)

