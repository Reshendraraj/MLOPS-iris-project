import joblib
import os
import sys # Import sys for logging configuration
import logging # Import logging module
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Basic logging configuration to output to console
# This is a fallback/debug setup. If src.logger is configured for console,
# this might be redundant but ensures output during development.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)


# Assuming get_logger is defined in src.logger and works correctly
# If not, you might need to adjust based on its implementation
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self):
        # Corrected path to match where DataProcessing saves files
        self.processed_data_path = "artifacts/raw"
        self.model_path = "artifacts/models"

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize the model
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)

        logger.info("Model Training Initialized.")
        logger.info(f"Processed data will be loaded from: {self.processed_data_path}")
        logger.info(f"Model will be saved to: {self.model_path}")


    def load_data(self):
        logger.info(f"Attempting to load data from {self.processed_data_path}")
        try:
            # Define full paths to the pickle files
            X_train_path = os.path.join(self.processed_data_path, "X_train.pkl")
            X_test_path = os.path.join(self.processed_data_path, "X_test.pkl")
            y_train_path = os.path.join(self.processed_data_path, "y_train.pkl")
            y_test_path = os.path.join(self.processed_data_path, "y_test.pkl")

            # Check if files exist before loading
            if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
                missing_files = [p for p in [X_train_path, X_test_path, y_train_path, y_test_path] if not os.path.exists(p)]
                logger.error(f"Missing processed data files: {missing_files}")
                raise FileNotFoundError(f"Missing processed data files: {missing_files}")

            X_train = joblib.load(X_train_path)
            X_test = joblib.load(X_test_path)
            y_train = joblib.load(y_train_path)
            y_test = joblib.load(y_test_path)

            logger.info("Data loaded successfully.")
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")

            return X_train, X_test, y_train, y_test

        except FileNotFoundError as e:
            logger.error(f"File not found error while loading data: {e}")
            raise CustomException("Error while loading data - files not found", e)
        except Exception as e:
            # Use logger.error for actual errors
            logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True) # exc_info=True logs traceback
            raise CustomException("An unexpected error occurred while loading data", e)


    def train_model(self, X_train, y_train):
        # Check if data is provided
        if X_train is None or y_train is None:
            logger.warning("Training data is not provided. Skipping model training.")
            return # Exit if no data

        logger.info("Starting model training...")
        try:
            self.model.fit(X_train, y_train)

            # Ensure model directory exists before saving
            os.makedirs(self.model_path, exist_ok=True)
            model_save_path = os.path.join(self.model_path, "model.pkl")
            joblib.dump(self.model, model_save_path)

            logger.info(f"Model trained and saved successfully to {model_save_path}")

        except Exception as e:
            logger.error(f"Error while model training: {e}", exc_info=True)
            raise CustomException("Error while model training", e)


    def evaluate_model(self, X_test, y_test):
        # Check if data is provided and model is trained
        if X_test is None or y_test is None:
            logger.warning("Evaluation data is not provided. Skipping model evaluation.")
            return # Exit if no data
        if self.model is None:
             logger.warning("Model is not trained. Skipping model evaluation.")
             return # Exit if model is not trained


        logger.info("Starting model evaluation...")
        try:
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            # Use zero_division=0 to handle cases where a class has no predicted samples
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            logger.info(f"Evaluation Metrics:")
            logger.info(f"  Accuracy Score : {accuracy:.4f}") # Format to 4 decimal places
            logger.info(f"  Precision Score : {precision:.4f}")
            logger.info(f"  Recall Score : {recall:.4f}")
            logger.info(f"  F1 Score : {f1:.4f}")

            cm = confusion_matrix(y_test, y_pred)

            # Plotting the confusion matrix
            plt.figure(figsize=(8, 6))
            # Get unique labels from both y_test and y_pred for comprehensive labels
            labels = np.unique(np.concatenate((y_test, y_pred)))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")

            # Ensure model directory exists before saving plot
            os.makedirs(self.model_path, exist_ok=True)
            confusion_matrix_path = os.path.join(self.model_path, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close() # Close the plot to free memory

            logger.info(f"Confusion Matrix saved successfully to {confusion_matrix_path}")

        except Exception as e:
            logger.error(f"Error while model evaluation: {e}", exc_info=True)
            raise CustomException("Error while model evaluation", e)


    def run(self):
        logger.info("Starting model training run.")
        X_train, X_test, y_train, y_test = None, None, None, None # Initialize variables

        try:
            # Load data first
            X_train, X_test, y_train, y_test = self.load_data()

            # Only proceed if data was loaded successfully
            if X_train is not None and y_train is not None:
                self.train_model(X_train, y_train)
            else:
                logger.error("Training data not loaded, skipping training and evaluation.")
                return # Exit run if training data isn't available

            # Only proceed to evaluate if test data is available and model is trained
            if X_test is not None and y_test is not None and self.model is not None:
                 self.evaluate_model(X_test, y_test)
            else:
                 logger.error("Evaluation data or trained model not available, skipping evaluation.")


        except CustomException as e:
            # Catch custom exceptions raised by the methods and log them
            logger.error(f"A custom exception occurred during run: {e}", exc_info=True)
            # Depending on desired behavior, you might re-raise the exception
            # raise e
        except Exception as e:
            # Catch any other unexpected exceptions during the run
            logger.critical(f"An unexpected error occurred during run: {e}", exc_info=True)
            # Depending on desired behavior, you might re-raise the exception
            # raise e


        logger.info("Model training run finished.")


if __name__ == "__main__":
    logger.info("Starting the main execution block for Model Training.")
    try:
        # Ensure the processed data directory exists before initializing
        processed_dir = "artifacts/raw" # Match the DataProcessing output directory
        if not os.path.exists(processed_dir):
            logger.error(f"Processed data directory not found: {processed_dir}")
            logger.info("Please run the data processing step first to generate the required files.")
            sys.exit(1) # Exit if the processed data directory doesn't exist

        trainer = ModelTraining()
        trainer.run()

    except Exception as e:
        # Catch any exceptions that weren't handled within the ModelTraining class
        logger.critical(f"Script terminated due to an unhandled exception: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero code to indicate an error

    logger.info("Main execution block for Model Training finished.")
