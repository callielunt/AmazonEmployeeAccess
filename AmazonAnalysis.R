# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ggplot2)

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

# Create Plots
ggplot(data = train) + geom_boxplot(aes(x = RESOURCE , y = ACTION))

# Dummy Variable encode all nominal predictors
# Combine categories that occur less than 0.1% of the time into an "other" category
my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_dummy(all_nominal_predictors())


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

## Logistic regression

library(tidymodels)

# Make model
logRegModel <- logistic_reg() %>% 
  set_engine("glm")

# put into workflow
logreg_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logRegModel) %>% 
  fit(data = train)

# make predictions
amazon_predictions <- predict(logreg_wf,
                              new_data = test,
                              type = "prob")

submission_logreg <- amazon_predictions |>
  rename(ACTION = .pred_1) %>% 
  select(ACTION) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, ACTION) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_logreg, file = "./LogReg.csv", delim = ",")
