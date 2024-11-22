
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vroom)
library(dbarts)
library(themis)
library(embed)

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")
train <- train %>% mutate(ACTION = as.factor(ACTION))

# Recipe
my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 3)

my_mod_bart <- parsnip::bart(trees = 1000,
                             prior_terminal_node_coef = 0.95,
                             prior_terminal_node_expo = 2.00,
                             prior_outcome_range = 2.00 ) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

#Workflow
bart_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod_bart)


## Finalize workflow and predict 
final_wf_bart <-
  bart_wf %>% 
  fit(data = train)


## Predict
predictions_bart <- final_wf_bart %>% predict(new_data = test, type="prob")


submission_bart <- predictions_bart |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind predictions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_bart, file = "bart.csv", delim = ",")


