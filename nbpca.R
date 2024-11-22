#### Naive Bayes ####

# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ggplot2)
library(doParallel)
library(discrim)


# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

# Recipe
my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.8)

# Model
## Define a model

## nb model
nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

## work flow
nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_model)

## Tuning
## Set up a grid of tuning values
grid_of_tuning_params_nb <- grid_regular(Laplace(),
                                         smoothness(),
                                         levels = 5)

## Set up K-fold CV
folds_nb <- vfold_cv(train, v = 5, repeats = 1)

## Find best tuning parameters
CV_results_nb <- nb_wf %>% 
  tune_grid(resamples = folds_nb,
            grid = grid_of_tuning_params_nb,
            metrics = metric_set(roc_auc))

bestTune_nb <- CV_results_nb %>% 
  select_best(metric = "roc_auc")

## Finalize workflow and predict 
final_wf_nb <-
  nb_wf %>% 
  finalize_workflow(bestTune_nb) %>% 
  fit(data = train)

# make predictions
amazon_predictions <- predict(final_wf_nb,
                              new_data = test,
                              type = "prob")

submission_nb <- amazon_predictions |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind predictions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_nb, file = "nbPCA.csv", delim = ",")

