import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier

from scripts.feature_engineering import Featurizer

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
    """ Handles dataset loading, preprocessing, scaling, and resampling. """
    def __init__(self, splits, featuriser):
        self.splits = splits
        self.featuriser = featuriser

    def load_featurised_dataset(self):
        """
        Load featurized datasets for training, validation, and testing.
        
        Reads CSV files corresponding to different dataset splits, removes 
        unnecessary columns ('key' and 'input'), and returns the processed datasets.
        
        Returns:
            train, validation, test dataframes
        """

        logger.info("Checking for featurized dataset files...")
        
        featurised_splits = [
            f"../data/{split}_{self.featuriser}_featurized.csv" 
            for split in self.splits
        ]
        
        # Check if all files exist
        missing_files = [file for file in featurised_splits if not os.path.exists(file)]
        
        if missing_files:
            logger.error("Featurized dataset not found. Please featurize the dataset first.")
            for file in missing_files:
                logger.error(f"Missing file: {file}")
            return None, None, None
        
        data_dict = {}
        
        for i, name in enumerate(self.splits):
            data_dict[name] = pd.read_csv(featurised_splits[i]).drop(columns=['key', 'input'])
            logger.info(f"{name.capitalize()} dataset loaded and processed.")
        
        train, validation, test = data_dict["train"], data_dict["validation"], data_dict["test"]
        return train, validation, test
        
    def get_scaled_set(self, train, test, val, target='Y'):
        """
        Scale the feature sets and return the processed training, validation, and test datasets.
    
        Args:
            train (DataFrame): Training dataset containing features and target.
            test (DataFrame): Testing dataset containing features and target.
            val (DataFrame): Validation dataset containing features and target.
            target (str): The column name of the target variable. Default is 'Y'.
    
        Returns:
            Scaled feature matrices and target variables for train, test, and validation sets.
        """
        logger.info("Scaling feature sets...")
    
        def clean_data(X, y):
            """Removes all-zero or NaN rows and updates y accordingly."""
            X = X[(X != 0).any(axis = 1) & X.notna().any(axis = 1)]
            y = y.loc[X.index]
            
            return X, y
    
        # Extract features and target
        X_train, y_train = clean_data(train.drop(columns = [target]), train[target])
        X_test, y_test = clean_data(test.drop(columns = [target]), test[target])
        X_val, y_val = clean_data(val.drop(columns = [target]), val[target])
    
        # Identify and remove zero-variance columns
        zero_columns = X_train.columns[X_train.nunique() == 1].tolist()
        X_train.drop(columns=zero_columns, inplace=True)
        X_test.drop(columns=zero_columns, inplace=True)
        X_val.drop(columns=zero_columns, inplace=True)
    
        logger.info(f"Removed {len(zero_columns)} zero-variance columns.")
    
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
    
        logger.info("Feature scaling completed successfully.")
    
        return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val

    
    def preprocess_and_resample(self, target='Y'):
        """
        Preprocess the dataset by loading, preparing features and targets, scaling, 
        and applying resampling techniques (SMOTE, Random Oversampling, and SMOTEENN).
        
        Args:
            target (str): The column name of the target variable. Default is 'Y'.
        
        Returns:
            Scaled and resampled datasets including:
            - Original train, test, and validation sets
            - SMOTE-resampled train set
            - Random oversampled train set
            - Hybrid (SMOTE + ENN) resampled train set
        """
        logger.info("Starting dataset preprocessing and resampling...")
        train, validation, test = self.load_featurised_dataset()

        if all(x is not None for x in [train, validation, test]):
        
            X_train, y_train, X_test, y_test, X_val, y_val = self.get_scaled_set(train, test, validation, target = target)
            logger.info("Feature scaling completed.")
            
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            logger.info("SMOTE resampling completed.")
            
            oversampler = RandomOverSampler(random_state=42) # Random Oversampling
            X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)
            logger.info("Random Oversampling completed.")
            
            smote_enn = SMOTEENN(random_state=42) # Hybrid (SMOTE + ENN)
            X_train_hybrid, y_train_hybrid = smote_enn.fit_resample(X_train, y_train)
            logger.info("SMOTE + ENN Hybrid Resampling completed.")
    
            logger.info("Preprocessing and resampling completed successfully.")

        else:
            X_train = y_train = X_test = y_test = X_val = y_val = X_train_smote = y_train_smote = X_train_over = y_train_over = X_train_hybrid = y_train_hybrid = None
            logger.info("Please featurize the dataset first.")

        return X_train, y_train, X_test, y_test, X_val, y_val, X_train_smote, y_train_smote, X_train_over, y_train_over, X_train_hybrid, y_train_hybrid

class Modelling:
    """A class for training, evaluating, and visualizing machine learning models (RandomForest and XGBoost)."""
    
    def train_model(self, model_type, X_train, y_train, X_test, y_test, X_val, y_val, 
                    class_weight=False, use_stratified_kfold=False, use_gridsearch=False):
        """
        Generalized function to train RandomForest or XGBoost classifier.
        
        Args:
            model_type: 'randomforest' or 'xgboost'
            X_train, y_train: Training features and labels
            X_test, y_test: Test features and labels
            X_val, y_val: Validation features and labels
            class_weight: Boolean, whether to apply class weighting
            use_stratified_kfold: Boolean, whether to use StratifiedKFold cross-validation
            use_gridsearch: Boolean, whether to perform GridSearchCV for hyperparameter tuning
    
        Returns:
            Tuple: (Trained model, Dictionary containing metrics and predictions)
        """
        logger.info(f"Initializing {model_type.capitalize()} Classifier...")
        
        if model_type == "randomforest":
            params = {
                "n_estimators": 200,
                "criterion": "gini",
                "min_samples_split": 5,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
                "max_depth" : 5
            }
            
            class_weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            if class_weight:
                params["class_weight"] = class_weight_dict
                params["max_depth"] = 3

                
            model = RandomForestClassifier(**params)
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "max_features": ["sqrt", "log2"]
            }
        
        elif model_type == "xgboost":
            params = {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 4 if class_weight else 3,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
            
            if class_weight:
                params["scale_pos_weight"] = sum(y_train == 0) / sum(y_train == 1)
            
            model = XGBClassifier(**params)
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.01],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.8],
                "colsample_bytree": [0.8, 1.0]
            }
        else:
            raise ValueError("Unsupported model type. Use 'randomforest' or 'xgboost'.")
        
        if use_gridsearch:
            logger.info("Performing GridSearchCV for hyperparameter tuning...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")

        
        elif use_stratified_kfold:
            logger.info("Performing StratifiedKFold cross-validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
                model.fit(X_t, y_t)
                y_pred_v = model.predict(X_v)
                cv_scores.append(roc_auc_score(y_v, y_pred_v))
            logger.info(f"Cross-Validation ROC AUC Scores: {cv_scores}")
            logger.info(f"Mean ROC AUC: {sum(cv_scores) / len(cv_scores):.4f}")

        else:
            model.fit(X_train, y_train)
        
        # Compute metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = self.compute_metrics(y_train, y_train_pred, y_train_proba, "Train")
        
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = self.compute_metrics(y_val, y_val_pred, y_val_proba, "Validation")
        
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = self.compute_metrics(y_test, y_test_pred, y_test_proba, "Test") 
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        
        return model, {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_train_pred": y_train_pred,
            "y_train_proba": y_train_proba,
            "y_val_pred": y_val_pred,
            "y_val_proba": y_val_proba,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve
        }


    def evaluate_model(self, model_results):
        """
        Evaluate multiple models and select the one with the highest ROC-AUC score.
        
        Parameters:
            model_results (dict): A dictionary where keys are model names and values are dictionaries
                              containing test metrics, including 'roc_auc'.
        
        Returns:
            Tuple: The best model and its corresponding results based on the highest ROC-AUC score.
        """
        best_roc_auc = 0
        model = None
        results = None
        for model, results in model_results.items():
            roc_score = results['test_metrics']['roc_auc']
            
            if roc_score > best_roc_auc:
                best_roc_auc = roc_score
                best_model = model
                best_results = results
        
        logger.info(f"Best Model: {best_model}, Best ROC-AUC Score: {best_roc_auc}")
        return model, best_results


    def visualize_model(self, model, model_results, graph_title, y_test):
        """
        Visualizes the performance of a given model using various metrics and plots.

        Parameters:
            - model: The trained machine learning model.
            - model_results (dict): Dictionary containing training, validation, and test metrics.
            - graph_title (str): Title for the plots.
            - y_test (array-like): True labels for the test set.

        Generates:
            - ROC Curve
            - Precision-Recall Curve
            - Confusion Matrix
            - Feature Importance (if applicable)
            - Prints tabular performance metrics
        """
        save_path = f"../data/figures/"
        
        train_metric = model_results['train_metrics'] 
        val_metric = model_results['val_metrics'] 
        test_metric = model_results['test_metrics']

        best_roc_auc = test_metric['roc_auc']
        
        logger.info(f"Best Model: {model}")
        logger.info(f"Best ROC-AUC Score: {best_roc_auc:.4f}")

        df = pd.DataFrame({'Train': train_metric, 'Validation': val_metric, 'Test': test_metric})

        # Print in tabular format
        print(tabulate.tabulate(df, headers = 'keys', tablefmt = 'grid'))

        fpr, tpr, _ = roc_curve(y_test, model_results["y_test_proba"])

        # ROC Curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {best_roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {graph_title}")
        plt.legend()
        plt.grid(True)
        roc_curve_path = os.path.join(save_path, f"ROC Curve - {graph_title}.png")
        plt.savefig(roc_curve_path)
        plt.show()
        logger.info(f"ROC Curve saved at {roc_curve_path}")

        precision_curve = model_results["precision_curve"]
        recall_curve = model_results["recall_curve"]
        
        # Precision-Recall Curve
        plt.figure(figsize=(6, 4))
        plt.plot(recall_curve, precision_curve, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {graph_title}.png")
        plt.legend()
        plt.grid(True)
        pr_curve_path = os.path.join(save_path, f"Precision-Recall Curve - {graph_title}.png")
        plt.savefig(pr_curve_path)
        plt.show()
        logger.info(f"Precision-Recall Curve saved at {pr_curve_path}")
        
        y_test_pred = model_results["y_test_pred"]
        cm = confusion_matrix(y_test, y_test_pred) 
        TN, FP, FN, TP = cm.ravel()
    
        specificity = TN / (TN + FP)
        npv = TN / (TN + FN) 
        
        logger.info(f"Specificity: {specificity:.4f}, Negative Predictive Value: {npv:.4f}")

        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Blockage", "Blockage"], yticklabels=["No Blockage", "Blockage"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {graph_title}\nSpecificity: {specificity:.2%}, NPV: {npv:.2%}")
        plt.grid(True)
        cm_path = os.path.join(save_path, f"Confusion Matrix - {graph_title}.png")
        plt.savefig(cm_path)
        plt.show()
        logger.info(f"Confusion Matrix saved at {cm_path}")

        # Feature_importances
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
            
            sorted_indices = np.argsort(feature_importances)[-10:]
            top_features = np.array(feature_names)[sorted_indices]
            top_importances = feature_importances[sorted_indices]
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x=top_features, y=top_importances)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance")
            plt.title(f"Top 10 Feature Importances: {graph_title}")
            plt.xticks(rotation = 45)
            plt.grid(True)
            feature_importance_path = os.path.join(save_path, f"Top 10 Feature Importances - {graph_title}.png")
            plt.savefig(feature_importance_path)
            plt.show()
            logger.info(f"Top 10 Feature Importances saved at {feature_importance_path}")
            logger.info(f"Top 10 Feature Importances: {list(zip(top_features, top_importances))}")

    def model_config(self, X_train, y_train, X_train_over, y_train_over, X_train_smote, y_train_smote, X_train_hybrid, y_train_hybrid):
        """
        Prepares different training datasets and model configurations for experimentation.

        Parameters:
            - X_train, y_train: Original training dataset.
            - X_train_over, y_train_over: Oversampled training dataset.
            - X_train_smote, y_train_smote: SMOTE-generated training dataset.
            - X_train_hybrid, y_train_hybrid: Hybrid resampled training dataset.

        Returns:
            - train_sets (dict): A dictionary containing different training datasets.
            - configs (list): A list of model configuration dictionaries.
        """
        train_sets = {
            "Original Set": (X_train, y_train),
            "Oversampled": (X_train_over, y_train_over),
            "SMOTE": (X_train_smote, y_train_smote),
            "Hybrid": (X_train_hybrid, y_train_hybrid)
            }
        
        configs = [
            {"class_weight": False, "use_stratified_kfold": False, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": False, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": True, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": False, "use_gridsearch": True},
            {"class_weight": False, "use_stratified_kfold": True, "use_gridsearch": False},
            {"class_weight": False, "use_stratified_kfold": False, "use_gridsearch": True},
            ]
        
        logger.info(f"Prepared {len(train_sets)} training sets: {list(train_sets.keys())}")
        logger.info(f"Generated {len(configs)} model configurations.")
    
        return train_sets, configs
        
    def compute_metrics(self, y_true, y_pred, y_proba, split_name):
        """
        Compute classification metrics and log results.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        roc_auc = roc_auc_score(y_true, y_proba)
        
        logger.info(f"{split_name} Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}, ROC AUC={roc_auc:.4f}")
        
        return {
                "confusion_matrix": (tn, fp, fn, tp),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc
            }
    
    def apply_trained_model(self, file, featuriser):
        """
        Apply a trained model to a given dataset, perform prediction, and evaluate model performance.
    
        Parameters:
            file : The name of the input file (without the file extension) that contains the dataset to be processed.
            featuriser : The identifier for the featurizer used to process the dataset (e.g., 'eos5guo' or 'eos2gw4').
        
        Returns: None
            Prints a table of performance metrics for the applied model, including Accuracy, Specificity, 
            Negative Predictive Value (NPV), and ROC-AUC score.
        """
        try:
            output_path = f'../data/{file}_{featuriser}_featurized.csv'
            
            logger.info("Applying the trained model to dataset...")
            if not os.path.exists(output_path): #Only featurise files that have not been previously featurised
                featurizer = Featurizer(model_id = featuriser)
                output_path = featurizer.featurize_csv(input_file = file)
            
            with open(f"../models/best_{featuriser}_model.pkl", "rb") as f:
                model = pkl.load(f)
    
            X = pd.read_csv(output_path)
            y = X['Y']
            X.drop(columns=['key', 'input', 'Y'], inplace = True)
            
            if featuriser == 'eos5guo':
                cols_to_drop = [
                        'dimension_179', 'dimension_180', 'dimension_181', 'dimension_218', 'dimension_219', 'dimension_220', 'dimension_221', 'dimension_222', 'dimension_223',
                        'dimension_235', 'dimension_236', 'dimension_295', 'dimension_296', 'dimension_297', 'dimension_298', 'dimension_299', 'dimension_300', 'dimension_301',
                        'dimension_302', 'dimension_303', 'dimension_304', 'dimension_305', 'dimension_306', 'dimension_307', 'dimension_308', 'dimension_309', 'dimension_310',
                        'dimension_311', 'dimension_312', 'dimension_313', 'dimension_314'
                    ]
                X.drop(columns=cols_to_drop, inplace=True)
                logger.info("Dropped extra dimensions for eos5guo featuriser")
            
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            accuracy = (tp + tn)/(tp + tn + fp + fn)
            specificity = tn / (tn + fp)  
            npv = tn / (tn + fn) 
            roc_auc = roc_auc_score(y, y_proba)
    
            # Tabulate results
            metrics_table = [
                ["Accuracy", accuracy],
                ["Specificity", specificity],
                ["Negative Predictive Value (NPV)", npv],
                ["ROC-AUC", roc_auc]
            ]
        
            print(tabulate.tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
        except Exception as e:
            logger.error(f"Error Applying Model: {e}")

    def make_predictions(self, X, featuriser):
        """
        Make predictions on a given dataset using a pre-trained model.
    
        Parameters:
            X : The featuris=zed dataset on which predictions will be made.
            featuriser : The identifier for the featurizer whose best model is to be loaded
    
        Returns: None
            Prints a table of predictions and probabilities for the input dataset, including the 
            predicted class ("Herg Blocker" or "Not Herg Blocker") and the associated probability.
        """
        try:
            logger.info("Making Prediction...")

            model_path = f"../models/best_{featuriser}_model.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return

            with open(model_path, "rb") as f:
                model = pkl.load(f)
                logger.info(f"Model loaded successfully from {model_path}")

            if featuriser == 'eos5guo':
                cols_to_drop = [
                        'dimension_179', 'dimension_180', 'dimension_181', 'dimension_218', 'dimension_219', 'dimension_220', 'dimension_221', 'dimension_222', 'dimension_223',
                        'dimension_235', 'dimension_236', 'dimension_295', 'dimension_296', 'dimension_297', 'dimension_298', 'dimension_299', 'dimension_300', 'dimension_301',
                        'dimension_302', 'dimension_303', 'dimension_304', 'dimension_305', 'dimension_306', 'dimension_307', 'dimension_308', 'dimension_309', 'dimension_310',
                        'dimension_311', 'dimension_312', 'dimension_313', 'dimension_314'
                    ]
                X.drop(columns=cols_to_drop, inplace=True)
                logger.info("Dropped extra dimensions for eos5guo featuriser")
                
            try:
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.error("Error while making prediction")
                return

            # Tabulate results
            metrics_table = [
                ["Prediction", "Herg Blocker" if y_pred == 1 else "Not Herg Blocker"],
                ["Probability", y_proba]
            ]
        
            print(tabulate.tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
