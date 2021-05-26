# load packages
library(tidymodels)
library(tidyverse)
library(lubridate)
library(ggplot2)

# set seed
set.seed(15)

# import data
loan_train <- read_csv("data/train.csv")
loan_test <- read_csv("data/test.csv")

# explore outcome variable
loan_train %>% 
  ggplot(aes(money_made_inv)) +
  geom_histogram(bins = 20)
  
loan_train %>% 
  ggplot(aes(money_made_inv)) +
  geom_density(color = "orange", fill = "orange", alpha = 0.5)

# money made inv seems left skewed 
# try log transform to normalize

# loan_train <- loan_train %>% 
#   mutate(
#     money_made_inv = abs(money_made_inv) 
#   ) %>% 
#   mutate(
#     money_made_inv = log(money_made_inv) 
#   )

# loan_train %>% 
#   ggplot(aes(money_made_inv)) +
#   geom_density(color = "orange", fill = "orange", alpha = 0.5)
# loan_train %>% 
#   ggplot(aes(money_made_inv)) +
#   geom_histogram(color = "orange", fill = "orange", alpha = 0.5, bins = 30)
# # somewhat more normal distribution

loan_folds <- vfold_cv(loan_train,  v = 3, repeats = 3)

loan_recipe <- recipe(money_made_inv ~ term + out_prncp_inv + int_rate + application_type +
                        loan_amnt + tot_coll_amt + annual_inc + avg_cur_bal + home_ownership + dti + grade, data = loan_train) %>%
  step_dummy(all_nominal_predictors()) %>% # one-hot encode all categorical predictors
  step_normalize(all_numeric_predictors()) %>%
  step_interact(money_made_inv ~ (.)^2)

#we canuse prep and juice to verify that our recipe is working and transforming our results as we want it to  
prep(loan_recipe) %>% 
  bake(new_data = NULL)

lm_model <- linear_reg() %>% 
  set_engine("lm")

lm_workflow <- workflow() %>% 
  add_recipe(loan_recipe) %>% 
  add_model(lm_model)

lm_fit_resamples <- fit_resamples(lm_workflow, loan_folds)
#collect_metrics(lm_fit)
lm_best <- lm_workflow %>%
  finalize_workflow(select_best(lm_fit_resamples, metric = "rmse"))

lm_fit <- fit(lm_workflow, loan_train)

lm_pred <- predict(lm_fit, loan_test) %>% 
  bind_cols(Id = loan_test$id) %>% 
  rename(Predicted = .pred)

lm_pred <- lm_pred[, c(2,1)]
write_csv(lm_pred, "output/lm_pred.csv")

