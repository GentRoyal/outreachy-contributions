import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
    def __init__(self, dataset_name, splits, featuriser):
        self.dataset_name = dataset_name
        self.splits = splits
        self.featuriser = featuriser

    def load_featurised_dataset(self):
        """
        Load featurized datasets for training, validation, and testing.
        
        Reads CSV files corresponding to different dataset splits, removes 
        unnecessary columns ('key' and 'input'), and returns the processed datasets.
        
        Returns:
            DataFrames: train, validation, test.
        """
        logger.info("Loading featurized dataset...")
        
        featurised_partition = [
            f"../data/{self.dataset_name}/{key}_{self.featuriser}_featurized.csv" 
            for key in self.splits.keys()
        ]
    
        featurised_partition = [
            f"../data/{self.dataset_name}/{key}_{self.featuriser}_featurized.csv" 
            for key in self.splits.keys()
        ]
        
        train = pd.read_csv(featurised_partition[0])
        train.drop(columns=['key', 'input'], inplace=True)
        logger.info("Train dataset loaded and processed.")
        
        validation = pd.read_csv(featurised_partition[1])
        validation.drop(columns=['key', 'input'], inplace=True)
        logger.info("Validation dataset loaded and processed.")
        
        test = pd.read_csv(featurised_partition[2])
        test.drop(columns=['key', 'input'], inplace=True)
        logger.info("Test dataset loaded and processed.")

        return train, validation, test

    def prepare_features_and_target(self, df, target='Y'):
        """
        Prepare feature matrix (X) and target variable (y) from the dataset.
        
        Args:
            df (DataFrame): The dataset containing features and target variable.
            target (str): The column name of the target variable. Default is 'Y'.
        
        Returns:
            DataFrame: Processed dataset with corrected column names and filtered rows.
        """
        logger.info("Preparing features and target variable...")
    
        X = df.drop(columns=[target])
        y = df[target]

        if self.featuriser == 'eos24ci': 
            # The last character in the last 10 columns in DrugTax Featurized sets makes the model throw exceptions. 
            # So,I replaced them
            new_cols = X.columns.tolist()
            xters = [chr(char) for char in range(ord('a'), ord('z') + 1)][:10]
            
            for i, (char, col) in enumerate(zip(X.columns[-10:], xters)):
                new_cols[-10 + i] = char[:-1] + col  # Replace only the last character
            
            X.columns = new_cols

        X = X[(X != 0).any(axis=1) & X.notna().any(axis=1)]
        y = y.loc[X.index]
        
        df = X.merge(y, left_on=X.index, right_on=y.index, how='inner').drop(columns=['key_0'])
        logger.info("Features and target variable prepared successfully.")
        
        return df
    
    def get_scaled_set(self, train, test, val, target='Y'):
        """
        Scale the feature sets and return the processed training, validation, and test datasets.
        
        Args:
            train (DataFrame): Training dataset containing features and target.
            test (DataFrame): Testing dataset containing features and target.
            val (DataFrame): Validation dataset containing features and target.
            target (str): The column name of the target variable. Default is 'Y'.
        
        Returns:
            tuple: Scaled feature matrices and target variables for train, test, and validation sets.
        """
        logger.info("Scaling feature sets...")
        X_train, y_train = train.drop(columns=[target]), train[target]    
        X_test, y_test = test.drop(columns=[target]), test[target]
        X_val, y_val = val.drop(columns=[target]), val[target]

        #zero_columns = X_train.columns[(X_train.nunique() == 1)]
        zero_columns = ['dimension_179', 'dimension_180', 'dimension_181', 'dimension_218', 'dimension_219', 'dimension_220', 'dimension_221', 'dimension_222', 'dimension_223', 
                        'dimension_235', 'dimension_236', 'dimension_295', 'dimension_296', 'dimension_297', 'dimension_298', 'dimension_299', 'dimension_300', 'dimension_301', 
                        'dimension_302', 'dimension_303', 'dimension_304', 'dimension_305', 'dimension_306', 'dimension_307', 'dimension_308', 'dimension_309', 'dimension_310', 
                        'dimension_311', 'dimension_312', 'dimension_313', 'dimension_314']
        X_train.drop(columns = zero_columns, inplace=True)
        X_test.drop(columns = zero_columns, inplace=True)
        X_val.drop(columns = zero_columns, inplace=True)

        logger.info(f"Removed {len(zero_columns)} zero-variance columns.")

        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)
        X_val_scaled = scalar.transform(X_val)

        logger.info("Feature scaling completed successfully.")

        return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val
    
    def preprocess_and_resample(self, target='Y'):
        """
        Preprocess the dataset by loading, preparing features and targets, scaling, 
        and applying resampling techniques (SMOTE, Random Oversampling, and SMOTEENN).
        
        Args:
            target (str): The column name of the target variable. Default is 'Y'.
        
        Returns:
            tuple: Processed and resampled datasets including:
                - Original train, test, and validation sets (scaled)
                - SMOTE-resampled train set
                - Random oversampled train set
                - Hybrid (SMOTE + ENN) resampled train set
        """
        logger.info("Starting dataset preprocessing and resampling...")
        train, validation, test = self.load_featurised_dataset()
        train = self.prepare_features_and_target(train, target=target)
        test = self.prepare_features_and_target(test, target=target)
        val = self.prepare_features_and_target(validation, target=target)

        logger.info("Feature preparation completed.")
        
        X_train, y_train, X_test, y_test, X_val, y_val = self.get_scaled_set(train, test, val, target=target)
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

        return (X_train, y_train, 
                X_test, y_test, 
                X_val, y_val, 
                X_train_smote, y_train_smote, 
                X_train_over, y_train_over, 
                X_train_hybrid, y_train_hybrid
               )

class Modelling:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def train_randomforest_cv(self, X_train, y_train, X_test, y_test, X_val, y_val, class_weight, use_stratified_kfold = False, use_gridsearch = False):
        """
        Train a RandomForest classifier with optional GridSearchCV and StratifiedKFold validation.
    
        Args:
            X_train, y_train: Training features and labels
            X_test, y_test: Test features and labels
            X_val, y_val: Validation features and labels
            class_weight: Boolean, whether to apply class weighting
            use_stratified_kfold: Boolean, whether to use StratifiedKFold cross-validation
            use_gridsearch: Boolean, whether to perform GridSearchCV for hyperparameter tuning
    
        Returns:
            Tuple: (Trained model, Dictionary containing metrics and predictions)
        """
        logger.info("Initializing RandomForest Classifier...")
        
        rf_params = {
            "n_estimators": 200,
            "criterion": "gini",
            "min_samples_split": 5,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1
        }
        
        class_weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
        rf_params["max_depth"] = 4 if class_weight else 3
        if class_weight:
            rf_params["class_weight"] = class_weight_dict
    
        rf = RandomForestClassifier(**rf_params)
    
        if use_gridsearch:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "max_features": ["sqrt", "log2"]
            }
    
            grid_search = GridSearchCV(rf, param_grid, cv = 5, scoring = "roc_auc", n_jobs = -1, verbose = 1)
            grid_search.fit(X_train, y_train)
            rf = grid_search.best_estimator_
            logger.info(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")

        if use_stratified_kfold:
            logger.info("Performing StratifiedKFold cross-validation...")
            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
            cv_scores = []
    
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
    
                rf.fit(X_t, y_t)
                y_pred_v = rf.predict(X_v)
                cv_scores.append(roc_auc_score(y_v, y_pred_v))
    
            logger.info(f"Cross-Validation ROC AUC Scores: {cv_scores}")
            logger.info(f"Mean ROC AUC: {sum(cv_scores) / len(cv_scores):.4f}")
    
        rf.fit(X_train, y_train)
        
        def compute_metrics(y_true, y_pred, y_proba, dataset_name):
            """
            Compute classification metrics and log results.
            """
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            roc_auc = roc_auc_score(y_true, y_proba)

            logger.info(f"{self.dataset_name} Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}, ROC AUC={roc_auc:.4f}")

            return {
                "confusion_matrix": (tn, fp, fn, tp),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc
            }
    
        # Compute metrics
        y_train_pred = rf.predict(X_train)
        y_train_proba = rf.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, "Train")
    
        y_val_pred = rf.predict(X_val)
        y_val_proba = rf.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, "Validation")
    
        y_test_pred = rf.predict(X_test)
        y_test_proba = rf.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test") 
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    
        return rf, {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_train_pred": y_train_pred,
            "y_train_proba": y_train_proba,
            "y_val_pred": y_val_pred,
            "y_val_proba": y_val_proba,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "precision_curve" : precision_curve, 
            "recall_curve" : recall_curve
        }

    def train_xgboost_cv(self, X_train, y_train, X_test, y_test, X_val, y_val, class_weight = None, use_stratified_kfold = False, use_gridsearch = False):
        """
        Train a XGBoost classifier with optional GridSearchCV and StratifiedKFold validation.
    
        Args:
            X_train, y_train: Training features and labels
            X_test, y_test: Test features and labels
            X_val, y_val: Validation features and labels
            class_weight: Boolean, whether to apply class weighting
            use_stratified_kfold: Boolean, whether to use StratifiedKFold cross-validation
            use_gridsearch: Boolean, whether to perform GridSearchCV for hyperparameter tuning
    
        Returns:
            Tuple: (Trained model, Dictionary containing metrics and predictions)
        """
        logger.info("Initializing XGBoost Classifier...")
        
        xgb_params = {
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
            xgb_params["scale_pos_weight"] = sum(y_train == 0) / sum(y_train == 1) 
        
        xgb = XGBClassifier(**xgb_params)
        
        if use_gridsearch:
            logger.info("Performing GridSearchCV for hyperparameter tuning...")

            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.01],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.8],
                "colsample_bytree": [0.8, 1.0]
            }
            
            grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            xgb = grid_search.best_estimator_
            logger.info(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
        
        if use_stratified_kfold:
            logger.info("Performing StratifiedKFold cross-validation...")

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
                
                xgb.fit(X_t, y_t)
                y_pred_v = xgb.predict(X_v)
                cv_scores.append(roc_auc_score(y_v, y_pred_v))
            
            logger.info(f"Cross-Validation ROC AUC Scores: {cv_scores}")
            logger.info(f"Mean ROC AUC: {sum(cv_scores) / len(cv_scores):.4f}")
        
        xgb.fit(X_train, y_train)
        
        def compute_metrics(y_true, y_pred, y_proba, dataset_name):
            """
            Compute classification metrics and log results.
            """
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            roc_auc = roc_auc_score(y_true, y_proba)
    
            logger.info(f"{self.dataset_name} Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}, ROC AUC={roc_auc:.4f}")
            
            return {
                "confusion_matrix": (tn, fp, fn, tp),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc
            }
    
        y_train_pred = xgb.predict(X_train)
        y_train_proba = xgb.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, "Train")
    
        y_val_pred = xgb.predict(X_val)
        y_val_proba = xgb.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, "Validation")
    
        y_test_pred = xgb.predict(X_test)
        y_test_proba = xgb.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test") 
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        
        return xgb, {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_train_pred": y_train_pred,
            "y_train_proba": y_train_proba,
            "y_val_pred": y_val_pred,
            "y_val_proba": y_val_proba,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "precision_curve" : precision_curve,
            "recall_curve" : recall_curve
            
        }

    def evaluate_model(self, model_results):
        best_roc_auc = 0
        model = None
        results = None
        for model, results in model_results.items():
            roc_score = results['test_metrics']['roc_auc']
            
            if roc_score > best_roc_auc:
                best_roc_auc = roc_score
                best_model = model
                best_results = results

        return model, best_results


    def visualize_model(self, model, model_results, graph_title, y_test):
        save_path = f"../data/figures/{self.dataset_name}/"

        train_metric = model_results['train_metrics']
        val_metric = model_results['val_metrics']
        test_metric = model_results['test_metrics']

        best_roc_auc = test_metric['roc_auc']
            
        
        print("Best Model:", model)
        print("Best ROC-Score:", best_roc_auc)

        df = pd.DataFrame({'Train': train_metric, 'Validation': val_metric, 'Test': test_metric})

        # Print in tabular format
        print(tabulate.tabulate(df, headers='keys', tablefmt='grid'))
        
        fpr, tpr, _ = roc_curve(y_test, model_results["y_test_proba"])
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {best_roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {graph_title}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"ROC Curve- {graph_title}.png"))
        plt.show()
        
        precision_curve = model_results["precision_curve"]
        recall_curve = model_results["recall_curve"]
        
        plt.figure(figsize=(6, 4))
        plt.plot(recall_curve, precision_curve, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {graph_title}.png")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"Precision-Recall Curve- {graph_title}.png"))
        plt.show()
        
        y_test_pred = model_results["y_test_pred"]
        cm = confusion_matrix(y_test, y_test_pred)
        TN, FP, FN, TP = cm.ravel()
    
        specificity = TN / (TN + FP)
        npv = TN / (TN + FN) 
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Blockage", "Blockage"], yticklabels=["No Blockage", "Blockage"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {graph_title}\nSpecificity: {specificity:.2%}, NPV: {npv:.2%}")
        plt.grid(True)
        # Save the figure
        plt.savefig(os.path.join(save_path, f"Confusion Matrix: {graph_title}"))
        plt.show()
    
    
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
            plt.savefig(os.path.join(save_path, f"Top 10 Feature Importances: {graph_title}"))
            plt.show()

    def model_config(self, X_train, y_train, X_train_over, y_train_over, X_train_smote, y_train_smote, X_train_hybrid, y_train_hybrid):
        train_sets = {
            "Original Set": (X_train, y_train),
            "Oversampled": (X_train_over, y_train_over),
            "SMOTE": (X_train_smote, y_train_smote),
            "Hybrid": (X_train_hybrid, y_train_hybrid),
            }
            
        configs = [
            {"class_weight": False, "use_stratified_kfold": False, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": False, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": True, "use_gridsearch": False},
            {"class_weight": True, "use_stratified_kfold": False, "use_gridsearch": True},
            {"class_weight": False, "use_stratified_kfold": True, "use_gridsearch": False},
            {"class_weight": False, "use_stratified_kfold": False, "use_gridsearch": True},
            ]
    
        return train_sets, configs

    def apply_trained_model(self, model, X_train, y_train, X_test, y_test, X_val, y_val):
        def compute_metrics(y_true, y_pred, y_proba, dataset_name):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            roc_auc = roc_auc_score(y_true, y_proba)
    
            # Print results inside the function
            print("=" * 50)
            print(f"{self.dataset_name} Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print("=" * 50)
    
            return {
                "confusion_matrix": (tn, fp, fn, tp),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc
            }
    
        # Compute metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, "Train")
    
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, "Validation")
    
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test") 
        
        # Precision-Recall curve
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
            "precision_curve" : precision_curve, 
            "recall_curve" : recall_curve
        }