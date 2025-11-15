from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# Constants
MODEL_PATH = "waste_collection_model.joblib"
SCALER_PATH = "feature_scaler.joblib"
DATASET_PATH = "historical_data.csv"
METRICS_PATH = "model_metrics.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WasteCollectionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_columns = [] # To store feature names used for training
        self.historical_data = pd.DataFrame(columns=['timestamp', 'fillLevel']) # Initialize empty
        self.load_or_initialize_model()

    def load_historical_data(self):
        """Loads historical data from CSV, ensures correct format."""
        if os.path.exists(DATASET_PATH):
            try:
                df = pd.read_csv(DATASET_PATH)
                # Convert timestamp column to datetime objects immediately
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp couldn't be parsed
                df = df.sort_values('timestamp').reset_index(drop=True) # Sort by time
                
                if 'timestamp' not in df.columns or 'fillLevel' not in df.columns:
                    raise ValueError("Invalid dataset format. Required columns: timestamp, fillLevel")
                
                logger.info(f"Loaded {len(df)} records from {DATASET_PATH}")
                return df
            except Exception as e:
                logger.error(f"Error loading historical data from {DATASET_PATH}: {e}")
        logger.info(f"No historical data found at {DATASET_PATH}. Starting with empty DataFrame.")
        return pd.DataFrame(columns=['timestamp', 'fillLevel'])

    def load_or_initialize_model(self):
        """Loads saved model and scaler, or initializes new ones."""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                
                # After loading, try to re-load historical data to ensure predictor has it
                self.historical_data = self.load_historical_data() 
                if not self.historical_data.empty:
                    # Re-create features on historical data to ensure feature_columns is set correctly
                    # This helps ensure consistency when predict is called after server restart.
                    temp_df = self.create_features(self.historical_data.copy(), is_training=True)
                    # We don't need X or y here, just feature_columns set by prepare_features
                    self.prepare_features(temp_df) 

                self.is_trained = True
                logger.info("Model and scaler loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Error loading model or scaler: {e}")
        self._initialize_new_model()
        
        # If no model was loaded, try to train with any existing historical data
        if not self.historical_data.empty:
            logger.info("Existing historical data found, attempting to train a new model.")
            self.train(self.historical_data)


    def _initialize_new_model(self):
        """Initializes a new Gradient Boosting Regressor and StandardScaler."""
        # Slightly more complex model, potentially better for richer features
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, subsample=0.7)
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("New model and scaler initialized.")

    def create_features(self, df, is_training=True):
        """
        Engineers time-based, rolling, and lagged features from the raw DataFrame.
        is_training: If True, NaNs from feature creation (e.g., beginning of series) are dropped.
                     If False (for prediction), NaNs are filled with 0 (or last known value).
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df = df.sort_values('timestamp').reset_index(drop=True) # Ensure sorted for rolling/lag features

        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_year'] = df['timestamp'].dt.dayofyear # New feature
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int) # New feature

        # --- Enhanced Lag Features ---
        # These are crucial for time series. For prediction, they will 'look back' into historical_data.
        df['fillLevel_lag_1'] = df['fillLevel'].shift(1)
        df['fillLevel_lag_2'] = df['fillLevel'].shift(2)
        df['fillLevel_lag_7'] = df['fillLevel'].shift(7) # Previous week's value on the same day

        # --- Enhanced Rolling Features ---
        # Calculate rolling features based on available 'fillLevel'.
        # min_periods=1 ensures they are calculated even if not enough data for full window.
        df['rolling_mean_3d'] = df['fillLevel'].rolling(window=3, min_periods=1).mean()
        df['rolling_std_3d'] = df['fillLevel'].rolling(window=3, min_periods=1).std()
        df['rolling_mean_7d'] = df['fillLevel'].rolling(window=7, min_periods=1).mean()
        df['rolling_std_7d'] = df['fillLevel'].rolling(window=7, min_periods=1).std()
        
        # Holiday feature (basic example, could integrate a proper holiday calendar)
        # For a full solution, you'd load a list of holidays and create a binary feature
        df['is_holiday_dummy'] = 0 # Placeholder: Implement proper holiday detection if needed
        # Example: if any of the dates were 2024-01-01 (New Year's Day), set to 1

        # Handle NaNs created by shift/rolling operations
        if is_training:
            # For training, it's best to drop rows that don't have full feature sets
            original_len = len(df)
            df.dropna(subset=self.get_all_feature_column_names(), inplace=True) # Drop if any essential feature is NaN
            if len(df) < original_len:
                logger.warning(f"Dropped {original_len - len(df)} rows due to NaNs during feature creation for training.")
        else:
            # For prediction, we must keep all future dates. Fill NaNs with a sensible value (e.g., 0 or last known value).
            # A more robust approach might be to fill NaNs in lagged features with the last observed value from historical_data
            # before the start of the prediction period. For now, 0 for missing values is simple.
            df.fillna(0, inplace=True) 
            
        return df

    def get_all_feature_column_names(self):
        """Returns a list of all feature column names expected by the model."""
        return [
            'day_of_week', 'day_of_month', 'month', 'is_weekend', 'day_of_year', 'week_of_year',
            'fillLevel_lag_1', 'fillLevel_lag_2', 'fillLevel_lag_7',
            'rolling_mean_3d', 'rolling_std_3d', 'rolling_mean_7d', 'rolling_std_7d',
            'is_holiday_dummy'
        ]

    def prepare_features(self, df):
        """Selects and orders feature columns for the model."""
        self.feature_columns = self.get_all_feature_column_names()
        # Ensure all columns exist, fill with 0 if somehow missing (shouldn't if create_features is robust)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_columns]

    def train(self, df):
        """Trains the model with new historical data."""
        try:
            self.historical_data = df.copy() # Update the stored historical data
            self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp']) # Ensure datetime type
            self.historical_data = self.historical_data.sort_values('timestamp').reset_index(drop=True)

            df_features = self.create_features(self.historical_data, is_training=True)
            
            if df_features.empty:
                logger.warning("No data to train on after feature creation. Model not trained.")
                self.is_trained = False
                return

            X = self.prepare_features(df_features)
            y = df_features['fillLevel']
            
            # Simple time-series train-validation split (last 20% for validation)
            # Ensure enough data points for a split
            if len(X) > 10: # Minimum data points to make a split meaningful
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                
                logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples.")

                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                self.model.fit(X_train_scaled, y_train)

                # Evaluate and save metrics
                val_predictions = self.model.predict(X_val_scaled)
                val_predictions = np.clip(val_predictions, 0, 100) # Clip for realistic metrics
                
                mae = mean_absolute_error(y_val, val_predictions)
                rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
                r2 = r2_score(y_val, val_predictions)

                metrics_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': 'Gradient Boosting Regressor',
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'R2': round(r2, 2),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
                metrics_df = pd.DataFrame([metrics_data])
                if os.path.exists(METRICS_PATH):
                    metrics_df.to_csv(METRICS_PATH, mode='a', header=False, index=False)
                else:
                    metrics_df.to_csv(METRICS_PATH, index=False)
                logger.info(f"Model trained. Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
            else:
                logger.warning("Not enough data for train/validation split. Training on all available data.")
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                # No validation metrics available in this case
                logger.info("Model trained on all available data.")

            self.is_trained = True
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info("Model and scaler saved successfully.")

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True) # Print traceback for debugging
            raise

    def predict(self, future_dates_str):
        """
        Generates predictions for future dates.
        future_dates_str: List of date strings ('YYYY-MM-DD').
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please upload data first.")
        
        if self.historical_data.empty:
            raise ValueError("No historical data available to make predictions. Please upload data.")

        # Convert future_dates_str to datetime objects
        future_dates_dt = [pd.to_datetime(d, errors='coerce') for d in future_dates_str]
        future_dates_dt = [d for d in future_dates_dt if pd.notna(d)] # Filter out invalid dates

        if not future_dates_dt:
            raise ValueError("No valid future dates provided for prediction.")

        # Create a DataFrame for future dates
        future_df = pd.DataFrame({'timestamp': future_dates_dt})
        future_df['fillLevel'] = np.nan # Fill level is unknown for future, use NaN as placeholder

        # Ensure historical_data is sorted by timestamp for correct feature generation
        self.historical_data = self.historical_data.sort_values('timestamp').reset_index(drop=True)

        # Combine historical data with future dates to correctly compute rolling/lag features.
        # The rolling window and lags will correctly extend from the end of the historical data.
        combined_df = pd.concat([self.historical_data, future_df], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        # Generate features for the combined dataframe.
        # `is_training=False` ensures NaNs are filled (not dropped) for future dates.
        combined_df_features = self.create_features(combined_df, is_training=False)
        
        # Extract only the features for the dates we want to predict
        # This involves filtering by timestamp to get only the future rows
        X_future_raw = combined_df_features[combined_df_features['timestamp'].isin(future_df['timestamp'])].copy()
        
        # Prepare features by selecting and ordering columns.
        X_future = self.prepare_features(X_future_raw)

        # Ensure X_future has the same number of features as X_train
        if X_future.shape[1] != self.scaler.n_features_in_:
             logger.error(f"Feature mismatch: Expected {self.scaler.n_features_in_} features, got {X_future.shape[1]}.")
             # Attempt to align columns if possible to prevent hard crash due to column order
             # This is a defensive measure, proper feature engineering should prevent this
             missing_cols = set(self.feature_columns) - set(X_future.columns)
             for c in missing_cols:
                 X_future[c] = 0
             X_future = X_future[self.feature_columns] # Reorder to match training order

        X_future_scaled = self.scaler.transform(X_future)
        
        predictions = self.model.predict(X_future_scaled)
        
        # Clip predictions to be within 0 and 100 percent
        predictions = np.clip(predictions, 0, 100) 
        
        response = []
        for i, date_dt in enumerate(future_df['timestamp']):
            response.append({
                'date': date_dt.strftime('%Y-%m-%d'),
                'predicted_fill_level': round(predictions[i], 2),
                # FIX: Convert boolean to string or explicit JSON compatible type if needed.
                # In most modern Flask/Werkzeug versions, this should implicitly convert to JSON boolean,
                # but if the error persists, consider: str(predictions[i] >= 75).lower()
                'should_collect': bool(predictions[i] >= 75) # Ensure it's a native Python bool
            })
        return response

# Initialize the predictor globally
predictor = WasteCollectionPredictor()

# --- Flask Routes (Remain largely the same, but will use improved predictor logic) ---

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def get_predictions():
    try:
        # Predict for 14 days starting from tomorrow
        future_dates_str = [(datetime.now() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, 15)]
        predictions = predictor.predict(future_dates_str)
        # The `predictor.predict` method now returns the correctly formatted list of dictionaries.
        # jsonify should handle a list of dictionaries with standard types (strings, numbers, booleans).
        return jsonify(predictions) 
    except ValueError as ve:
        logger.warning(f"Prediction pre-check failed: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Internal server error during prediction: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        # Read the uploaded CSV
        df = pd.read_csv(file)
        
        # Validate columns
        if 'timestamp' not in df.columns or 'fillLevel' not in df.columns:
            return jsonify({'error': 'Invalid CSV format. Required columns: timestamp, fillLevel'}), 400
        
        # Convert timestamp and sort before saving
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save the new historical data (overwriting previous)
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"New historical data saved to {DATASET_PATH} with {len(df)} records.")
        
        # Retrain the model with the new data
        predictor.train(df)
        
        return jsonify({'message': 'File uploaded and model retrained successfully'}), 200
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Uploaded CSV file is empty.'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Could not parse CSV file. Ensure it is well-formed.'}), 400
    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

@app.route('/dataset/info', methods=['GET'])
def dataset_info():
    """Returns dataset details like record count and date range"""
    try:
        # Ensure historical_data is up-to-date, especially after a fresh server start
        predictor.historical_data = predictor.load_historical_data() 

        if predictor.historical_data.empty:
            return jsonify({'total_records': 0, 'date_range': {'start': None, 'end': None}, 'is_trained': False}), 200
        
        # Ensure timestamp is datetime type for min/max
        predictor.historical_data['timestamp'] = pd.to_datetime(predictor.historical_data['timestamp'])

        start_date = predictor.historical_data['timestamp'].min().strftime('%Y-%m-%d')
        end_date = predictor.historical_data['timestamp'].max().strftime('%Y-%m-%d')
        
        return jsonify({
            'total_records': len(predictor.historical_data),
            'date_range': {'start': start_date, 'end': end_date},
            'is_trained': predictor.is_trained
        })
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def model_metrics():
    """Returns model performance metrics"""
    if not os.path.exists(METRICS_PATH):
        return jsonify({'error': 'No metrics available. Train the model first.'}), 400
    try:
        metrics_df = pd.read_csv(METRICS_PATH)
        metrics = metrics_df.to_dict(orient='records')
        return jsonify({'metrics': metrics})
    except Exception as e:
        logger.error(f"Error reading model metrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)