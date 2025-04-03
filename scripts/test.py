import pandas as pd
import pickle as pkl
import os
import sys
from glob import glob
from typing import Dict, Tuple, List, Optional, Any, Union

from load_dataset import DataDownloader
from clean_transform_data import DataProcessor
from exploratory_analysis import ExploratoryDataAnalysis
from feature_engineering import Featurizer
from train_evaluate_model import DatasetProcessor, Modelling

class DrugAnalysisApp:
    def __init__(self):
        self.datasets = ["hERG"]
        self.featurisers = ["eos2gw4", "eos5guo"]
        self.current_dataset = None
        self.current_featuriser = None
        self.splits = None
        self.df = None
        self.best_model = None
        
    def display_menu(self):
        """Display the main menu options"""
        print("\n===== DRUG ANALYSIS APPLICATION =====")
        print("1. Load Dataset")
        print("2. Featurize Dataset")
        print("3. Generate Exploratory Data Analysis")
        print("4. Apply Trained Model")
        print("5. Predict hERG Blocker Status")
        print("6. Exit")
        return input("\nSelect an option (1-6): ")
    
    def load_dataset(self):
        """Load a dataset - download new or use existing"""
        print("\n===== LOAD DATASET =====")
        
        # Select dataset
        while True:
            print("\nAvailable datasets:")
            for i, dataset in enumerate(self.datasets, 1):
                print(f"{i}. {dataset}")
            
            try:
                choice = int(input("\nSelect dataset number: "))
                if 1 <= choice <= len(self.datasets):
                    self.current_dataset = self.datasets[choice - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Download new or use existing
        while True:
            download_choice = input("\nUse existing data? (Y/N): ").strip().upper()
            if download_choice in ['Y', 'N']:
                break
            print("Invalid input. Please enter 'Y' or 'N'.")
            
        if download_choice == 'N':
            print(f"\nDownloading {self.current_dataset} dataset...")
            downloader = DataDownloader()
            self.df, self.splits = downloader.fetch_dataset(name = self.current_dataset)
            print(f"Dataset {self.current_dataset} successfully downloaded.")
        else:
            print(f"\nLoading existing {self.current_dataset} dataset...")
            try:
                self.df = pd.read_csv(f'../data/{self.current_dataset}.csv')
                train = pd.read_csv(f'../data/train.csv') 
                validation = pd.read_csv(f'../data/validation.csv') 
                test = pd.read_csv(f'../data/test.csv') 
                self.splits = {"train": train, "validation": validation, "test": test}
                print("Data successfully loaded.")
            except FileNotFoundError as e:
                print(f"Error: {e}. Dataset not found, attempting to download...")
                downloader = DataDownloader()
                self.df, self.splits = downloader.fetch_dataset(name = self.current_dataset)
        
        # Preprocessing
        files_to_preprocess = [f"../data/{key}.csv" for key in self.splits.keys()]
        for file in files_to_preprocess:
            processor = DataProcessor(input_csv = file, output_csv = file)
            processor.process_csv()
        
        print(f"\n{self.current_dataset} dataset loaded and preprocessed successfully.")
    
    def featurize_dataset(self):
        """Featurize the loaded dataset"""
        if not self.current_dataset:
            print("\nError: No dataset loaded. Please load a dataset first.")
            return
            
        print("\n===== FEATURIZE DATASET =====")
        
        # Select featuriser
        while True:
            print("\nAvailable featurizers:")
            for i, featuriser in enumerate(self.featurisers, 1):
                print(f"{i}. {featuriser}")
            
            try:
                choice = int(input("\nSelect featurizer number: "))
                if 1 <= choice <= len(self.featurisers):
                    self.current_featuriser = self.featurisers[choice - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        count = len(glob(f"../data/*{self.current_featuriser}_featurized*"))
        
        if count < 3:
            print(f"\nFeaturizing {self.current_dataset} dataset with {self.current_featuriser}...")
            featurizer = Featurizer(model_id = self.current_featuriser)
            files = ['train', 'test', 'validation']
            
            for file in files:
                output_path = featurizer.featurize(input_file = file)
            print("Featurization completed successfully.")
        else:
            print(f"\nDataset has already been featurized with {self.current_featuriser}.")
    
    def generate_eda(self):
        """Generate exploratory data analysis for the loaded dataset"""
        try:
            if not self.current_dataset:
                raise FileNotFoundError("No dataset loaded. Please load a dataset first.")
        
            print(f"\nGenerating exploratory data analysis for {self.current_dataset}...")
            eda = ExploratoryDataAnalysis(dataset_name=self.current_dataset)
            eda.generate_eda()
            print("EDA generation completed. Results are saved in the output directory.")
        
        except FileNotFoundError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
    
    def apply_trained_model(self):
        """Apply a trained model to the current dataset"""
        if not self.current_featuriser:
            print("\nError: Select a featurizer first.")
            return
            
        print("\n===== APPLY TRAINED MODEL =====")
        
        try:
            file_name = input("Enter a filename: [# Ensure file is located in `data` folder]")
            modelling = Modelling() 
            modelling.apply_trained_model(file_name, self.current_featuriser)
            
        except Exception as e:
            print(f"\nError applying trained model: {e}")
        
    def predict_herg_blocker(self):
        """Predict if a drug is a hERG blocker or not"""
        if not self.best_model:
            print("\nError: No trained model available. Please apply a trained model first.")
            return
            
        print("\n===== PREDICT hERG BLOCKER STATUS =====")
        
        # Here you would implement functionality to:
        # 1. Allow the user to input SMILES or drug name
        # 2. Featurize the input
        # 3. Apply the trained model to predict hERG blocker status
        
        print("\nThis functionality is coming soon.")
        
        # Placeholder for implementation
        smiles = input("\nEnter SMILES string for the drug: ")
        if not smiles:
            print("No SMILES provided. Returning to main menu.")
            return
            
        print(f"\nAnalyzing: {smiles}")
        print("\nPrediction: Not a hERG blocker (probability: 0.23)")
        print("\nNote: This is a placeholder result. Actual prediction functionality is coming soon.")
    
    def run(self):
        """Run the application main loop"""
        print("Welcome to the Drug Analysis Application!")
        
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.load_dataset()
            elif choice == '2':
                self.featurize_dataset()
            elif choice == '3':
                self.generate_eda()
            elif choice == '4':
                self.apply_trained_model()
            elif choice == '5':
                self.predict_herg_blocker()
            elif choice == '6':
                print("\nExiting application. Goodbye!")
                sys.exit(0)
            else:
                print("\nInvalid option. Please try again.")
                
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    app = DrugAnalysisApp()
    app.run()