# ===============================================
# Generates:
#   1. preproc_params.rds (Quantile rules)
#   2. dmy_transformer.rds (Encoder)
#   3. best_model.rds (Trained XGBoost Model)
# ===============================================

# --- 1. SETUP AND LIBRARIES ---
cat("--- 1. SETUP AND LIBRARIES ---\n")

# Load essential libraries
library(tidyverse)
library(caret)
library(rpart)
library(janitor)

# Libraries for Enhanced Modeling
library(glmnet)
library(ranger)
library(xgboost) # Added for the best model
library(doParallel)
library(pROC) # Required for ROC curves

# Set a global seed for reproducibility
set.seed(40)

# Setup parallel backend for speed
tryCatch({
  cl <- makePSOCKcluster(detectCores() - 1)
  registerDoParallel(cl)
  cat(paste("✅ Registered parallel backend with", detectCores() - 1, "cores.\n"))
}, error = function(e) {
  cat("Could not start parallel cluster. Running in sequential mode.\n")
})


# --- 2. DATA CLEANING ---
cat("\n--- 2. DATA CLEANING ---\n")

# Load the dataset
tryCatch({
  data <- read_csv("ai4i2020.csv", show_col_types = FALSE)
}, error = function(e) {
  cat("\n--- ERROR: ai4i2020.csv not found! ---\n")
  stop(e)
})

data_cleaned <- data %>%
  clean_names() %>%
  select(-udi, -product_id) %>% # Drop unique identifiers
  mutate(
    machine_failure = factor(machine_failure, levels = c(0, 1), labels = c("No", "Yes")),
    type = as.factor(type)
  ) %>%
  # Drop the data leakage flags
  select(-twf, -hdf, -pwf, -osf, -rnf)

cat("Data cleaned. Target 'machine_failure' converted to factor.\n")

# --- 2.1 EXPLORATORY DATA ANALYSIS (EDA) ---
cat("\n--- 2.1 GENERATING EDA PLOTS ---\n")

# A. Data Imbalance Plot
p1 <- ggplot(data_cleaned, aes(x = machine_failure, fill = machine_failure)) +
  geom_bar(alpha = 0.8) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  theme_minimal() +
  labs(title = "Class Distribution: Machine Failure",
       subtitle = "Visualizing Data Imbalance",
       x = "Machine Failure",
       y = "Count")

print(p1)
cat("Generated Data Imbalance Plot.\n")

# B. Box Plots for Numerical Features
# We pivot the data to 'long' format to plot all features at once with faceting
numeric_features <- data_cleaned %>%
  select(machine_failure, rotational_speed_rpm, torque_nm, 
         tool_wear_min, air_temperature_k, process_temperature_k) %>%
  pivot_longer(cols = -machine_failure, names_to = "Feature", values_to = "Value")

p2 <- ggplot(numeric_features, aes(x = machine_failure, y = Value, fill = machine_failure)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~Feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  theme_bw() +
  labs(title = "Feature Distributions by Target Class",
       subtitle = "Boxplots identifying outliers and separation",
       x = "Machine Failure",
       y = "Value")

print(p2)

# --- 3. TIME-SERIES DATA SPLIT (RAW DATA) ---
cat("\n--- 3. TIME-SERIES SPLIT ---\n")
# We must split the raw data first to prevent data leakage

split_index <- floor(nrow(data_cleaned) * 0.70)
train_raw <- data_cleaned[1:split_index, ]
test_raw <- data_cleaned[(split_index + 1):nrow(data_cleaned), ]

cat(paste("Raw data split: 70% Training (", nrow(train_raw), " rows), 30% Testing (", nrow(test_raw), " rows)\n"))


# --- 4. CREATE & SAVE PREPROCESSING PARAMETERS ---
cat("\n--- 4. CREATING & SAVING PREPROC PARAMETERS ---\n")
# Calculate parameters *only* from the training data

preproc_params <- list(
  q_torque_95 = quantile(train_raw$torque_nm, 0.95, na.rm = TRUE),
  q_speed_05 = quantile(train_raw$rotational_speed_rpm, 0.05, na.rm = TRUE),
  q_wear_95 = quantile(train_raw$tool_wear_min, 0.95, na.rm = TRUE)
)

# ✅ SAVE THE PARAMETERS FILE
saveRDS(preproc_params, "preproc_params.rds")
cat("✅ Successfully saved 'preproc_params.rds'\n")
print(preproc_params)


# --- 5. FEATURE ENGINEERING ---
cat("\n--- 5. FEATURE ENGINEERING ---\n")

# This helper function must be IDENTICAL to the one in the API
create_feature_df <- function(data, params) {
  features_raw <- data %>%
    select(
      type,
      air_temperature_k,
      process_temperature_k,
      rotational_speed_rpm,
      torque_nm,
      tool_wear_min
    )
  
  features_engineered <- features_raw %>%
    mutate(
      # Physics-Based Features
      power_kw = torque_nm * rotational_speed_rpm / 1000,
      temp_diff_k = process_temperature_k - air_temperature_k,
      torque_speed_ratio = torque_nm / (rotational_speed_rpm + 1e-6),
      
      # Binning
      tool_wear_category = cut(tool_wear_min,
                               breaks = c(-Inf, 50, 150, 250, Inf),
                               labels = c("New", "Moderate", "Worn", "Critical"),
                               right = TRUE),
      
      # Operational Extremes (Using LOADED quantiles)
      is_high_torque = torque_nm > params$q_torque_95,
      is_low_speed = rotational_speed_rpm < params$q_speed_05,
      is_extreme_wear = tool_wear_min > params$q_wear_95
    ) %>%
    # Drop the original column that was binned/replaced
    select(-tool_wear_min)  
  
  return(features_engineered)
}

# Apply feature engineering to both sets
train_features <- create_feature_df(train_raw, preproc_params)
test_features <- create_feature_df(test_raw, preproc_params)

cat("Feature engineering complete for train and test sets.\n")


# --- 6. CREATE & SAVE ENCODER ---
cat("\n--- 6. CREATING & SAVING ENCODER ---\n")

# Create the dummy variable recipe *only* from the training features
dmy_transformer <- dummyVars("~ .", data = train_features, fullRank = TRUE)

# ✅ SAVE THE TRANSFORMER FILE
saveRDS(dmy_transformer, "dmy_transformer.rds")
cat("✅ Successfully saved 'dmy_transformer.rds'\n")

# Apply the transformer to both sets
train_transformed <- predict(dmy_transformer, newdata = train_features)
test_transformed <- predict(dmy_transformer, newdata = test_features)

# Combine and sanitize for modeling
training <- as.data.frame(train_transformed) %>%
  mutate(across(where(is.integer), as.numeric)) %>%
  cbind(machine_failure = train_raw$machine_failure)

testing <- as.data.frame(test_transformed) %>%
  mutate(across(where(is.integer), as.numeric)) %>%
  cbind(machine_failure = test_raw$machine_failure)

cat("Data encoding and final prep complete.\n")


# --- 7. MODEL TRAINING ---
cat("\n=== 7. MODEL TRAINING (LR, DT, XGB) START ===\n")

# We apply SMOTE *inside* the CV loop to prevent data leakage
# We optimize for "Sens" (Sensitivity/Recall) to find the rare failures
ctrl_production <- trainControl(
  method = "repeatedcv",
  number = 5,     
  repeats = 2,    
  classProbs = TRUE,
  summaryFunction = twoClassSummary, 
  savePredictions = "final",
  sampling = "up",            # <--- CHANGE "smote" TO "up" HERE
  allowParallel = TRUE        
)
cat("TrainControl set up to use SMOTE and optimize for Sensitivity (Recall).\n")

# Define the model formula
formula <- machine_failure ~ .

# 7.1 MODEL 1: LOGISTIC REGRESSION (glmnet)
cat("\n--- 7.1 Training Regularized Logistic Regression (glmnet) ---\n")
model_glmnet <- train(formula, data = training,
                      method = "glmnet",
                      trControl = ctrl_production,
                      metric = "Sens", 
                      preProcess = c("center", "scale"), 
                      tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 0.1, length = 5)))

# 7.2 MODEL 2: DECISION TREE (rpart)
cat("\n--- 7.2 Training Tuned Decision Tree (rpart) ---\n")
model_dt_tuned <- train(formula, data = training,
                        method = "rpart",
                        trControl = ctrl_production,
                        metric = "Sens", 
                        tuneLength = 10)

# 7.3 MODEL 3: XGBOOST (Best Model)
cat("\n--- 7.3 Training XGBoost (xgbTree) ---\n")
tuneGrid_xgb <- expand.grid(
  nrounds = c(100, 150),
  max_depth = c(4, 6),
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)
model_xgb <- train(formula, data = training,
                   method = "xgbTree",
                   trControl = ctrl_production,
                   metric = "Sens", 
                   tuneGrid = tuneGrid_xgb,
                   verbose = 0)

cat("=== MODEL TRAINING COMPLETE ===\n")


# --- 8. MODEL EVALUATION & SAVING BEST MODEL ---
cat("\n=== 8. MODEL EVALUATION & SAVING ===\n")

models_list <- list(
  GLMNET = model_glmnet,
  DT_Tuned = model_dt_tuned,
  XGBoost = model_xgb
)

# Get CV results
resamps <- resamples(models_list)
print(summary(resamps))

# Evaluate on Test Set to find the best model
test_results <- data.frame(
  Model = names(models_list),
  Test_F1 = numeric(length(models_list)),
  Test_Sens = numeric(length(models_list)),
  Test_Prec = numeric(length(models_list)),
  Test_Acc = numeric(length(models_list))
)

for(i in 1:length(models_list)) {
  model <- models_list[[i]]
  model_name <- names(models_list)[i]
  pred_class <- predict(model, testing)
  
  cm <- confusionMatrix(pred_class, testing$machine_failure, positive = "Yes", mode = "everything")
  
  test_results$Test_Sens[i] <- cm$byClass["Sensitivity"]
  test_results$Test_Prec[i] <- cm$byClass["Precision"]
  test_results$Test_F1[i] <- cm$byClass["F1"]
}

cat("\n\n--- Final Test Set Performance Summary ---")
print(
  test_results %>%
    mutate(across(where(is.numeric), ~ round(.x, 4)))
)

# --- 8.1 ADVANCED VISUALIZATIONS ---
cat("\n--- 8.1 GENERATING MODEL EVALUATION PLOTS ---\n")

# C. ROC-AUC Comparison Graph
cat("Generating ROC Curves...\n")

# Prepare a data frame to store ROC data for plotting
roc_plot_data <- data.frame()

# Loop through models to calculate ROC
for(model_name in names(models_list)) {
  # Get probability predictions (focus on the "Yes" class)
  probs <- predict(models_list[[model_name]], testing, type = "prob")[, "Yes"]
  
  # Calculate ROC object using pROC
  roc_obj <- roc(testing$machine_failure, probs, quiet = TRUE)
  
  # Extract sensitivity (TPR) and specificity (1 - FPR)
  temp_df <- data.frame(
    Specificity = roc_obj$specificities,
    Sensitivity = roc_obj$sensitivities,
    Model = paste0(model_name, " (AUC: ", round(auc(roc_obj), 3), ")")
  )
  roc_plot_data <- rbind(roc_plot_data, temp_df)
}

# Plot ROC Curves
p3 <- ggplot(roc_plot_data, aes(x = 1 - Specificity, y = Sensitivity, color = Model)) +
  geom_path(size = 1) +
  geom_abline(linetype = "dashed", color = "gray") + # Random guess line
  theme_minimal() +
  labs(title = "ROC - AUC Comparison",
       subtitle = "Model Performance on Test Set",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme(legend.position = "bottom")

print(p3)


# D. Confusion Matrices for Each Model
cat("Generating Confusion Matrix Heatmaps...\n")

# Function to plot a single Confusion Matrix
plot_cm <- function(model, model_name, data) {
  pred <- predict(model, data)
  cm <- confusionMatrix(pred, data$machine_failure, positive = "Yes")
  
  # Convert CM table to data frame
  cm_df <- as.data.frame(cm$table)
  
  # Plot
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "gray70", high = "#0072B2") +
    labs(title = paste("Confusion Matrix:", model_name),
         subtitle = paste("Acc:", round(cm$overall['Accuracy'], 3), 
                          "| Sens:", round(cm$byClass['Sensitivity'], 3)),
         x = "Actual Class",
         y = "Predicted Class") +
    theme_minimal()
}

# Create and print CM plots for all models
cm_plots <- list()
for(name in names(models_list)) {
  print(plot_cm(models_list[[name]], name, testing))
}

cat("✅ Visualizations complete.\n")

# Find and save the best model based on F1 Score
best_model_name <- test_results$Model[which.max(test_results$Test_F1)]
best_model <- models_list[[best_model_name]]

cat(paste("\n🏆 BEST MODEL (by F1-Score) IS:", best_model_name), "\n")

# SAVE THE BEST MODEL
saveRDS(best_model, "best_model.rds")
cat("Successfully saved 'best_model.rds'\n")

cat("\n--- PIPELINE COMPLETE ---
All 3 artifacts (preproc_params.rds, dmy_transformer.rds, best_model.rds) are saved.
You can now run your Plumber API and then run the index.html file to open a server.\n")

# Stop parallel cluster
tryCatch({
  stopCluster(cl)
  cat("Parallel cluster stopped.\n")
}, error = function(e) {})