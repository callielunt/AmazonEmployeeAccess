#####
# Penalized Logistic Regression
#####

# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ggplot2)
library(doParallel)

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

# Make model
logPenRegModel <- logistic_reg(mixture = tune() , penalty = tune()) %>% 
  set_engine("glmnet")

# Workflow
penreg_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logPenRegModel)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- penreg_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best(metric = "roc_auc")

## Finalize workflow
final_wf <-
  penreg_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = train)

# make predictions
amazon_predictions <- predict(final_wf,
                              new_data = test,
                              type = "prob")

submission_penlogreg <- amazon_predictions |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_penlogreg, file = "./PenLogReg.csv", delim = ",")

stopCluster(cl)


