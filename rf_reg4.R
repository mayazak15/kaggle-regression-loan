# load packages
library(tidymodels)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(naniar)

# set seed
set.seed(15)

tidymodels_prefer()


# import/ tidy data
loan_train <- read_csv("data/train.csv") %>% 
  mutate_if(is.character, as.factor)
loan_test <- read_csv("data/test.csv")

# explore correlations
loan_train %>%
  select(-c(where(is.factor), where(is.character), id)) %>%
  cor(use = "pairwise") %>%
  corrplot::corrplot(method = "circle")

#  set up cross validaiton with repeats
loan_folds <- vfold_cv(loan_train,  v = 5, repeats = 3)

# set up recipe
loan_recipe <- recipe(money_made_inv ~ out_prncp_inv + loan_amnt + acc_now_delinq + num_tl_120dpd_2m + avg_cur_bal + tot_cur_bal + num_tl_30dpd + dti + int_rate + application_type
                      + tot_coll_amt + term + home_ownership + annual_inc + grade, data = loan_train) %>%
  step_dummy(all_nominal_predictors())

#we canuse prep and juice to verify that our recipe is working and transforming our results as we want it to  
prep(loan_recipe) %>% 
  bake(new_data = NULL)

# Define model ----
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 15)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 10)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe)

# Tuning/fitting ----
rf_tuned <- rf_workflow %>% 
  tune_grid(loan_folds, grid = rf_grid)

# chose workflow with best rmse
rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

# save results for fast access
save(rf_tuned, rf_workflow, file = "saving-models/rf4.4_tuned.rda")

# fit training data to the best workflow
rf_results <- fit(rf_workflow_tuned, loan_train)

# make predictions for testing data
# organize results for kaggle
rf_pred <- predict(rf_results, loan_test) %>% 
  bind_cols(Id = loan_test$id) %>% 
  rename(Predicted = .pred)
rf_pred <- rf_pred[, c(2,1)]

# write out results
write_csv(rf_pred, "output/rf4.4_pred.csv")
