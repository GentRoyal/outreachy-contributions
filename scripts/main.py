import pandas as pd
import pickle as pkl
import tabulate
import os
from glob import glob
from load_dataset import DataDownloader
from clean_transform_data import DataProcessor
from exploratory_analysis import ExploratoryDataAnalysis
from feature_engineering import Featurizer
from train_evaluate_model import DatasetProcessor, Modelling

#name = 'hERG' #input("Dataset name: ")
while True:
    name = input("Dataset name (AMES or hERG): ").strip()
    if name in ["AMES", "hERG"]:
        break
    print("Invalid input. Please enter either 'AMES' or 'hERG'.")

featuriser = "eos5guo" #ErG 2D
print("Starting Automation...")

# Step 1: Download Data
download = 'Y' #input("Use Existing Data? (Y/N): ").strip().upper()

splits = None

if download == 'N':
    print("\nDownloading data...")
    downloader = DataDownloader()
    df, splits = downloader.fetch_dataset(name = name)
else:
    print("\nLoading existing data...")
    try:
        df = pd.read_csv(f'../data/{name}/{name}.csv')
        train = pd.read_csv(f'../data/{name}/train.csv') 
        validation = pd.read_csv(f'../data/{name}/validation.csv') 
        test = pd.read_csv(f'../data/{name}/test.csv') 
        splits = {"train": train, "validation": validation, "test": test}
        print("Data successfully loaded.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Dataset not found, attempting to download...")
        downloader = DataDownloader()
        df, splits = downloader.fetch_dataset(name = name)


# Step 2: Data Preprocessing
print("\nPerforming Preprocessing...")
files_to_preprocess = [f"../data/{name}/{key}.csv" for key in splits.keys()]
for file in files_to_preprocess:
    processor = DataProcessor(input_csv = file, output_csv = file)
    processor.process_csv()


# Step 3: Perform EDA
print("\nPerforming EDA...")
eda = ExploratoryDataAnalysis(dataset_name = name)
eda.generate_eda()


# Step 4: Perform Featurisation
count = len(glob(f"../data/{name}/*{featuriser}_featurized*"))
print("\nPerforming Featurisation...")
if count == 0:
    featurizer = Featurizer(model_id = featuriser)
    files = ['train', 'test', 'validation']
    for file in files:
        output_path = featurizer.featurize(input_file = file, dataset_name = name)

else:
    print("\nData has already by Featurised...")


# Step 5: Modelling
data_processing = DatasetProcessor(dataset_name = name, splits = splits, featuriser = featuriser)
modelling = Modelling(dataset_name = name)

(X_train, y_train, 
X_test, y_test, 
X_val, y_val, 
X_train_smote, y_train_smote, 
X_train_over, y_train_over, 
X_train_hybrid, y_train_hybrid) = data_processing.preprocess_and_resample()

train_sets, configs = modelling.model_config(X_train, y_train, 
                                             X_train_over, y_train_over, 
                                             X_train_smote, y_train_smote, 
                                             X_train_hybrid, y_train_hybrid)

    
with open('../models/herg_model.pkl', "rb") as f:
    all_model_results = pkl.load(f)

best_roc_auc = 0
best_model = None
results = None
for model, results in all_model_results.items():
    roc_score = results['test_metrics']['roc_auc']
    
    if roc_score > best_roc_auc:
        best_roc_auc = roc_score
        best_model = model
        best_results = results

if name != 'hERG':
    train_sets, configs = modelling.model_config(X_train, y_train, 
            X_train_over, y_train_over, 
            X_train_smote, y_train_smote, 
            X_train_hybrid, y_train_hybrid)
    
    model_results = {}
    
    for train_set_name, (X_train_set, y_train_set) in train_sets.items():
        print(train_set_name)
        model, model_result = modelling.apply_trained_model(best_model, X_train_set, y_train_set, X_test, y_test, X_val, y_val)
        model_results[model] = model_result
    
    best_model, best_results = modelling.evaluate_model(model_results)

modelling.visualize_model(best_model, best_results, graph_title = "ERG 2D Featurized Dataset", y_test = y_test)


print("\nAutomation complete! All results are saved.")