# ===============================================
# AI4I 2020 - PLUMBER API
# v4.2 - Production Ready (Fixed Training-Serving Skew)
# ===============================================

library(plumber)
library(tidyverse)
library(caret)
library(xgboost)
library(ranger)
library(glmnet)

# --- ROBUST CORS FIX ---
# This "hook" handles the browser's "pre-flight" OPTIONS request
# which is sent before the actual POST request.
#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*") # Allow any website
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    # This is a pre-flight request. Tell the browser what's allowed.
    res$setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    res$setHeader("Access-Control-Allow-Headers", req$HTTP_ACCESS_CONTROL_REQUEST_HEADERS)
    res$status <- 200
    return(list())
  } else {
    # This is a regular request. Pass it on.
    plumber::forward()
  }
}
# --- END OF FIX ---


# --- 1. LOAD MODEL AND PREPROCESSOR ---
# These files MUST be in the same directory.
cat("Loading model...\n")
best_model <- readRDS("best_model.rds")
cat("Loading preprocessor...\n")
dmy_transformer <- readRDS("dmy_transformer.rds")

# ✅ CRITICAL FIX: Load the preprocessor parameters
cat("Loading preprocessor parameters...\n")
tryCatch({
  preproc_params <- readRDS("preproc_params.rds")
}, error = function(e) {
  stop("FATAL ERROR: 'preproc_params.rds' not found. 
       Please run the full 'AI4I_Production_Pipeline.R' script to generate it.")
})
cat("API models loaded. Ready to listen.\n")


# --- 2. HELPER FUNCTION (Feature Engineering) ---
# This is the *exact* same function from our final Shiny app
# to ensure the logic is identical.
create_feature_df <- function(input_data) {
  
  # Convert from user-friendly C to Kelvin
  air_temp_k <- as.numeric(input_data$air_temp) + 273.15
  process_temp_k <- as.numeric(input_data$process_temp) + 273.15
  
  # Get raw numeric values
  torque_val <- as.numeric(input_data$torque)
  rpm_val <- as.numeric(input_data$rpm)
  wear_val <- as.numeric(input_data$wear)
  
  # Create a single-row tibble with the raw inputs
  features_raw <- tibble(
    type = input_data$type,
    air_temperature_k = air_temp_k,
    process_temperature_k = process_temp_k,
    rotational_speed_rpm = rpm_val,
    torque_nm = torque_val,
    tool_wear_min = wear_val # Use the original column name
  )
  
  # Re-create the 7 *real* engineered features
  features_engineered <- features_raw %>%
    mutate(
      # 4.1 Physics-Based Features
      power_kw = torque_nm * rotational_speed_rpm / 1000,
      temp_diff_k = process_temperature_k - air_temperature_k,
      torque_speed_ratio = torque_nm / (rotational_speed_rpm + 1e-6),
      
      # 4.2 Binning
      tool_wear_category = cut(tool_wear_min,
                               breaks = c(-Inf, 50, 150, 250, Inf),
                               labels = c("New", "Moderate", "Worn", "Critical"),
                               right = TRUE),
      
      # 4.4 Operational Extremes (Using LOADED quantiles)
      # ✅ CRITICAL FIX: Use loaded params, not hard-coded numbers
      is_high_torque = torque_nm > preproc_params$q_torque_95,
      is_low_speed = rotational_speed_rpm < preproc_params$q_speed_05,
      is_extreme_wear = tool_wear_min > preproc_params$q_wear_95
    ) %>%
    # Drop the original column that was binned/replaced
    select(-tool_wear_min)  
  
  return(features_engineered)
}

#* @apiTitle Predictive Maintenance API
#* @apiDescription An API that uses our trained XGBoost model to predict machine failure.

#* Get a prediction
#* @param type Machine Type (L, M, or H)
#* @param air_temp Air Temperature in Celsius
#* @param process_temp Process Temperature in Celsius
#* @param rpm Rotational Speed in rpm
#* @param torque Torque in Nm
#* @param wear Tool Wear in minutes
#* @post /predict
function(type = "L", air_temp = 25, process_temp = 35, rpm = 1500, torque = 40, wear = 10) {
  
  # 1. Create a 1-row data frame from all inputs
  input_data <- data.frame(
    type = type,
    air_temp = air_temp,
    process_temp = process_temp,
    rpm = rpm,
    torque = torque,
    wear = wear
  )
  
  # 2. Run our helper function to do all 7 feature engineering steps
  features_to_predict <- create_feature_df(input_data)
  
  # 3. Apply the one-hot encoding (dmy) transformation
  final_data_point <- predict(dmy_transformer, newdata = features_to_predict)
  
  # 4. Get the prediction probability
  pred_probs <- predict(best_model, newdata = final_data_point, type = "prob")
  
  # 5. Get the risk percentage for "Yes"
  risk_percent <- round(pred_probs$Yes * 100, 1)
  
  # 6. Return the risk AND the feature logic as a JSON object
  # This lets the HTML dashboard display the *server's* calculations.
  return(list(
    risk = risk_percent,
    power_kw = features_to_predict$power_kw,
    temp_diff_k = features_to_predict$temp_diff_k,
    torque_speed_ratio = features_to_predict$torque_speed_ratio,
    is_high_torque = features_to_predict$is_high_torque,
    is_low_speed = features_to_predict$is_low_speed,
    is_extreme_wear = features_to_predict$is_extreme_wear
  ))
}