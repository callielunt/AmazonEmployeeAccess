# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")

my_recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_dummy(all_nominal_predictors())


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)
