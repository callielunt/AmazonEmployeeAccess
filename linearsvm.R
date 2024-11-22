# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)


# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")
train <- train %>% mutate(ACTION = as.factor(ACTION))

# Recipe
my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.9)



## SVM Models
svmLinear <- svm_linear(cost = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

## Fit and tune
# Workflow
lin_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svmLinear)

## Grid of values to tune over
tuning_grid_lin <- grid_regular(cost(),
                                levels = 5)


## Run the CV
CV_results_lin <- lin_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_lin,
            metrics = metric_set(roc_auc))

## Find best tuning parameters
bestTune_lin <- CV_results_lin %>% 
  select_best(metric = "roc_auc")

## Finalize workflow
final_wf_lin <-
  radial_workflow %>% 
  finalize_workflow(bestTune_lin) %>% 
  fit(data = train)

# make predictions
lin_predictions <- predict(final_wf_lin,
                           new_data = test,
                           type = "prob")

submission_lin <- lin_predictions |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_lin, file = "./linsvmpt1.csv", delim = ",")

