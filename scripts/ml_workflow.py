import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

matplotlib.use('Agg')  # This is correct for non-interactive backends

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger  =  logging.getLogger(__name__)

class HergBlocker:
    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name  =  dataset_name

    def load_featurized_dataset(self, model_name: str):
        """
        Load featurized train, validation, and test datasets.
        Args:
            model_name (str): The featurization method used.
        Returns:
            tuple: DataFrames for train, validation, and test sets.
        """
        base_path  =  Path("../data") / self.dataset_name
        file_paths  =  {split: base_path / f"{split}_{model_name}_featurized.csv" for split in ["train", "validation", "test"]}

        datasets  =  {}

        for split, path in file_paths.items():
            if not path.exists():
                logger.error(f"Missing dataset file: {path}")
                raise FileNotFoundError(f"Missing dataset file: {path}")

            logger.info(f"Loading {split} dataset from {path}")
            df  =  pd.read_csv(path)

            columns_to_drop  =  [col for col in ['key', 'input'] if col in df.columns]
            if columns_to_drop:
                df.drop(columns = columns_to_drop, inplace = True)
                logger.warning(f"Dropped columns {columns_to_drop} from {split} dataset")

            datasets[split] = df

        logger.info("All datasets loaded successfully.")
        return datasets["train"], datasets["validation"], datasets["test"]
        
    def prepare_features_and_target(df, target = 'Y', model_name = None):
        """
        Prepares features and target variable for machine learning.
        Args:
        df (pd.DataFrame): The dataset containing features and target.
        target (str, optional): The target column name. Defaults to 'Y'.
        model_name (str, optional): Model name for special processing. Defaults to None.
        Returns:
        pd.DataFrame: Processed dataset with cleaned features and target.
        """
        
        if target not in df.columns:
            logger.error(f"Target column '{target}' not found in dataset.")
            raise KeyError(f"Target column '{target}' not found.")
                
        X = df.drop(columns = [target])
        y = df[target]
        logger.info(f"Separated features (X) and target (y). X shape: {X.shape}, y shape: {y.shape}")
        
        if model_name == 'eos24ci':
            if X.shape[1] > =  10:
                new_cols  =  X.columns.tolist()
                xters  =  [chr(char) for char in range(ord('a'), ord('z') + 1)][:10]
                
                for i in range(10):
                    new_cols[-10 + i]  =  new_cols[-10 + i][:-1] + xters[i]
                    X.columns  =  new_cols
            logger.info("Renamed last 10 feature columns for 'eos24ci' model.")
        nan_rows  =  X.isna().sum(axis = 1) > 0
        X = X[(X !=  0).any(axis = 1) & ~nan_rows]
        y = y.loc[X.index]
        
        logger.info(f"Removed rows with NaNs or all-zero features. New shape: {X.shape}")
        
        df = X.merge(y, left_index = True, right_index = True, how = "inner")
        logger.info(f"Final dataset shape after merging features and target: {df.shape}")
            
        return df
        
    def get_scaled_set(train, test, val, target='Y'):
        """
        Scales train, test, and validation datasets after removing zero-variance columns.
        Args:
            train (pd.DataFrame): Training dataset.
            test (pd.DataFrame): Test dataset.
            val (pd.DataFrame): Validation dataset.
            target (str, optional): The target column. Defaults to 'Y'.
            
        Returns:
        tuple: Scaled feature sets and target variables (X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val).
        """
        
        for dataset, name in zip([train, test, val], ["train", "test", "val"]):
            if target not in dataset.columns:
                logger.error(f"Target column '{target}' not found in {name} set.")
                raise KeyError(f"Target column '{target}' not found in {name} set.")
            
                
        X_train, y_train = train.drop(columns=[target]).copy(), train[target].copy()
        X_test, y_test = test.drop(columns=[target]).copy(), test[target].copy()
        X_val, y_val = val.drop(columns=[target]).copy(), val[target].copy()
            
        zero_columns = X_train.columns[X_train.nunique() == 1].tolist()
        if zero_columns:
            logger.info(f"Removing zero-variance columns: {zero_columns}")
            X_train.drop(columns=zero_columns, inplace=True)
            X_test.drop(columns=[col for col in zero_columns if col in X_test.columns], inplace=True)
            X_val.drop(columns=[col for col in zero_columns if col in X_val.columns], inplace=True)
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
            
        logger.info("Feature scaling completed successfully.")
    
        return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val


    def preprocess_and_resample(model_name, target='Y'):
        """
        Loads, preprocesses, scales, and applies multiple resampling techniques to a dataset.
        Args:
            model_name (str): Model name for loading the dataset.
            target (str, optional): Target variable name. Defaults to 'Y'.
        Returns:
            tuple: Preprocessed and resampled datasets.
        """
        try:
            train, validation, test = load_featurised_dataset(model_name)
            logger.info("Dataset successfully loaded.")
    
            train = prepare_features_and_target(train, target=target)
            test = prepare_features_and_target(test, target=target)
            val = prepare_features_and_target(validation, target=target)
            logger.info("Feature preparation completed.")
    
            X_train, y_train, X_test, y_test, X_val, y_val = get_scaled_set(train, test, val, target=target)
            logger.info("Feature scaling completed.")
    
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied. Training set size increased to {X_train_smote.shape[0]}.")
    
            oversampler = RandomOverSampler(random_state=42)
            X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)
            logger.info(f"Random Oversampling applied. Training set size increased to {X_train_over.shape[0]}.")
    
            smote_enn = SMOTEENN(random_state=42)
            X_train_hybrid, y_train_hybrid = smote_enn.fit_resample(X_train, y_train)
            logger.info(f"SMOTE + ENN applied. Training set size adjusted to {X_train_hybrid.shape[0]}.")
    
            return (
                X_train, y_train, 
                X_test, y_test, 
                X_val, y_val, 
                X_train_smote, y_train_smote, 
                X_train_over, y_train_over, 
                X_train_hybrid, y_train_hybrid
            )
    
        except Exception as e:
            logger.error(f"Error in preprocess_and_resample: {str(e)}")
            raise


    def train_randomforest_cv(X_train, y_train, X_test, y_test, X_val, y_val, class_weight = True, use_stratified_kfold = False, use_gridsearch = False):
        """
        Train a RandomForest model with optional cross-validation and hyperparameter tuning.
        Args:
            X_train, y_train: Training set
            X_test, y_test: Test set
            X_val, y_val: Validation set
            class_weight (bool): Whether to apply class weighting
            use_stratified_kfold (bool): Use Stratified K-Fold cross-validation
            use_gridsearch (bool): Use GridSearchCV for hyperparameter tuning
        Returns:
            RandomForest model and performance metrics
        """
        
        rf_params = {
            "n_estimators": 200,
            "criterion": "gini",
            "min_samples_split": 5,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
    
        if class_weight:
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
            rf_params["class_weight"] = class_weight_dict
            logger.info(f"Using class weights: {class_weight_dict}")
    
        rf = RandomForestClassifier(**rf_params)
    
        if use_gridsearch:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 5, 6],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "max_features": ["sqrt", "log2"]
            }
    
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            rf = grid_search.best_estimator_
            logger.info(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
    
        if use_stratified_kfold:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
    
            for train_idx, val_idx in cv.split(np.array(X_train), np.array(y_train)):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
    
                rf.fit(X_t, y_t)
                y_pred_v = rf.predict(X_v)
                auc_score = roc_auc_score(y_v, y_pred_v)
                cv_scores.append(auc_score)
    
            logger.info(f"Cross-Validation ROC AUC Scores: {cv_scores}")
            logger.info(f"Mean ROC AUC: {sum(cv_scores) / len(cv_scores):.4f}")
    
        rf.fit(X_train, y_train)
        logger.info("Random Forest model training completed.")
    
        def compute_metrics(y_true, y_pred, y_proba, dataset_name):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_proba)
    
            logger.info(f"\n{dataset_name} Metrics:\n"
                        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                        f"Recall: {recall:.4f} | F1 Score: {f1:.4f} | ROC AUC: {roc_auc:.4f}")
    
            return {
                "confusion_matrix": (tn, fp, fn, tp),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
    
        y_train_pred, y_train_proba = rf.predict(X_train), rf.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, "Train")
    
        y_val_pred, y_val_proba = rf.predict(X_val), rf.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, "Validation")
    
        y_test_pred, y_test_proba = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    
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
            "precision_curve": precision_curve, 
            "recall_curve": recall_curve,
            "roc_curve": (fpr, tpr)
            }
