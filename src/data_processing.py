import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os
import sys # Import sys to configure logging to console
import logging # Import logging module

# Basic logging configuration to output to console
# You might have a more sophisticated setup in your src/logger.py
# but this ensures output for debugging if the custom logger isn't configured for console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)


# Assuming get_logger is defined in src.logger and works correctly
# If not, you might need to adjust based on its implementation
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None # Initialize df as None
        self.processed_data_path = "artifacts/raw" # Corrected typo 'artificats' to 'artifacts'
        # Ensure the directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)
        logger.info(f"Initialized DataProcessing with file_path: {self.file_path}")
        logger.info(f"Processed data will be saved to: {self.processed_data_path}")


    def load_data(self):
        logger.info(f"Attempting to load data from: {self.file_path}")
        try:
            # Corrected: Assign the DataFrame to self.df, not self.file_path
            self.df = pd.read_csv(self.file_path)
            logger.info("Data read successfully.")
            logger.info(f"Loaded DataFrame shape: {self.df.shape}")
            logger.info(f"Loaded DataFrame columns: {self.df.columns.tolist()}")

        except FileNotFoundError:
             logger.error(f"Error: File not found at {self.file_path}")
             raise CustomException(f"Failed to load data: File not found at {self.file_path}", FileNotFoundError)
        except Exception as e:
            # Use logger.error for actual errors
            logger.error(f"Error while reading data: {e}", exc_info=True) # exc_info=True logs traceback
            raise CustomException(f"Failed to load data from {self.file_path}", e)

    def handle_outliers(self, column):
        if self.df is None:
            logger.warning("DataFrame is not loaded. Cannot handle outliers.")
            return # Exit if no data is loaded

        if column not in self.df.columns:
             logger.warning(f"Column '{column}' not found in DataFrame. Cannot handle outliers.")
             return # Exit if column does not exist

        logger.info(f"Handling outliers for column: {column}")
        try:
            # Check if the column is numeric before calculating quantiles
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                logger.warning(f"Column '{column}' is not numeric. Skipping outlier handling.")
                return

            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            Lower_value = Q1 - 1.5 * IQR
            Upper_value = Q3 + 1.5 * IQR
            # Calculate median only if needed
            # sepal_median = np.median(self.df[column].dropna()) # Use dropna() to handle potential NaNs

            # More efficient way to replace outliers using boolean indexing and np.where
            # Replace values outside the bounds with the median of the column (excluding NaNs)
            median_value = self.df[column].median() # Use pandas median, handles NaNs by default
            self.df[column] = np.where(
                (self.df[column] < Lower_value) | (self.df[column] > Upper_value),
                median_value,
                self.df[column]
            )

            logger.info(f"Handled outliers successfully for column: {column}")
            logger.info(f"Outlier bounds for '{column}': Lower={Lower_value}, Upper={Upper_value}")


        except Exception as e:
            logger.error(f"Error while handling outliers for column '{column}': {e}", exc_info=True)
            raise CustomException(f"Failed to handle outliers for column '{column}'", e)

    def split_data(self):
        if self.df is None:
            logger.warning("DataFrame is not loaded. Cannot split data.")
            return # Exit if no data is loaded

        # Check if required columns exist
        required_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            logger.error(f"Missing required columns for splitting: {missing}")
            raise CustomException(f"Missing required columns for splitting: {missing}")

        logger.info("Splitting data...")
        try:
            X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            Y = self.df["Species"]

            # Ensure X and Y are not empty after selection
            if X.empty or Y.empty:
                 logger.error("Features or target are empty after selection.")
                 raise CustomException("Features or target are empty after selection.")

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            logger.info("Data split successfully.")
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")


            # Ensure the output directory exists before saving
            os.makedirs(self.processed_data_path, exist_ok=True)

            joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info(f"Split data files saved successfully to {self.processed_data_path}")

        except Exception as e:
            logger.error(f"Error while splitting or saving data: {e}", exc_info=True)
            raise CustomException("Failed to split or save data", e)

    def run(self):
        logger.info("Starting data processing run.")
        try:
            self.load_data()
            # Only proceed if data was loaded successfully
            if self.df is not None:
                self.handle_outliers("SepalWidthCm")
                self.split_data()
            else:
                logger.error("Data loading failed, skipping subsequent steps.")

        except CustomException as e:
            # Catch custom exceptions raised by the methods and log them
            logger.error(f"A custom exception occurred during run: {e}", exc_info=True)
            # Re-raise or handle as needed, depending on desired behavior
            # raise e # Uncomment to re-raise the exception after logging
        except Exception as e:
            # Catch any other unexpected exceptions during the run
            logger.error(f"An unexpected error occurred during run: {e}", exc_info=True)
            # raise e # Uncomment to re-raise the exception after logging

        logger.info("Data processing run finished.")


if __name__ == "__main__":
    # Ensure the 'artifacts/raw' directory exists before trying to instantiate
    # This prevents potential issues if the directory creation in __init__ somehow fails
    initial_data_path = "artifacts/raw/data.csv"
    artifacts_dir = os.path.dirname(initial_data_path)
    os.makedirs(artifacts_dir, exist_ok=True)

    # You might want to add a check here to see if data.csv exists
    if not os.path.exists(initial_data_path):
        logger.error(f"Input data file not found: {initial_data_path}")
        logger.info("Please ensure 'data.csv' is placed in the 'artifacts/raw' directory.")
        sys.exit(1) # Exit the script if the file is not found

    logger.info("Starting the main execution block.")
    try:
        data_processor = DataProcessing(initial_data_path)
        data_processor.run()
    except Exception as e:
        # Catch any exceptions that weren't handled within the DataProcessing class
        logger.critical(f"Script terminated due to an unhandled exception: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero code to indicate an error

    logger.info("Main execution block finished.")
