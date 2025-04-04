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

class hERGBlockerApp:
    def __init__(self):
        self.datasets = ["hERG"] #Can Expand to include AMES, etc.
        self.featurisers = ["eos2gw4", "eos5guo"]
        self.current_dataset = self.datasets[0]
        self.current_featuriser = None
        self.splits = None
        self.df = None
        self.best_model = None
        
    def display_menu(self):
        """Display the main menu options with improved design and instructions"""
        print("\n" + "=" * 50)
        print("             hERG BLOCKER PREDICTION SYSTEM")
        print("=" * 50)
        print(" NOTE: All input files should be placed inside the 'data' folder.")
        print("=" * 50)
        print(" INSTRUCTIONS:")
        print(" You can featurize compounds using any of the following:")
        print("  - eos5gu0  ->  ERG 2D Featurizer")
        print("  - eos2gw4  ->  Ersilia Compound Embeddings")
        print(" The best-performing model for each featurizer can be used to generate")
        print(" predictions or assess performance on unseen data.")
        print("=" * 50)
        print(" 1. Download Dataset")
        print(" 2. Featurize Dataset")
        print(" 3. Fill in Missing Drug Names")
        print(" 4. Generate Exploratory Data Analysis")
        print(" 5. Assess Performance on Unseen Data")
        print(" 6. Predict hERG Blocker Status")
        print(" 7. Exit")
        print("=" * 50)
        return input(" Please select an option (1-7): ")

    def load_dataset(self):
        """Download a dataset"""
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
        
        print(f"\nDownloading {self.current_dataset} dataset...")
        downloader = DataDownloader()
        self.df, self.splits = downloader.fetch_dataset(name = self.current_dataset)
        
    def clean_drug_id(self):
        """Fill in missing drug id's in a dataset"""
        file_name = input("Enter filename: ")
        file_path = os.path.join('../data/', f'{file_name}.csv')
        processor = DataProcessor(input_csv = file_path, output_csv = file_path)
        processor.process_csv()
    
    def featurize_dataset(self):
        """Featurize a dataset"""
        print("\n===== FEATURIZE DATASET =====")

        file_name = input("Enter filename: ")
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
        
        featurizer = Featurizer(model_id = self.current_featuriser)
        
        output_path = featurizer.featurize_csv(input_file = file_name)
    
    def generate_eda(self):
        """Generate exploratory data analysis for hERG dataset"""
        try:
            print(f"\nGenerating exploratory data analysis for {self.current_dataset}...")
            eda = ExploratoryDataAnalysis(dataset_name = self.current_dataset)
            eda.generate_eda()
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
    
    def apply_trained_model(self):
        """Apply a trained model to an unseen data"""
        # Select a featurizer to load the best model
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
            
        print("\n===== APPLY TRAINED MODEL =====")
        
        #try:
        file_name = input("Enter a filename: ")
        modelling = Modelling() 
        modelling.apply_trained_model(file_name, self.current_featuriser)
            
        #except Exception as e: print(f"\nError applying trained model: {e}")
        
    def predict_herg_blocker(self):
        """Predict if a drug is a hERG blocker or not"""
        
        print("\n===== PREDICT hERG BLOCKER STATUS =====")
        
        smiles = input("\nEnter SMILES string for the drug: ")
        
        if not smiles:
            print("No SMILES provided. Returning to main menu.")
            return

        # Select a featurizer to load the best model
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
        
        featurizer = Featurizer(model_id = self.current_featuriser)
        X = featurizer.featurize_smiles(smiles)
        modelling = Modelling()
        prediction = modelling.make_predictions(X, self.current_featuriser)
        
    def run(self):
        """Application main loop"""
        print("hERG Blocker Prediction Application!")
        
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.load_dataset()
            elif choice == '2':
                self.featurize_dataset()
            elif choice == '3':
                self.clean_drug_id()
            elif choice == '4':
                self.generate_eda()
            elif choice == '5':
                self.apply_trained_model()
            elif choice == '6':
                self.predict_herg_blocker()
            elif choice == '7':
                print("\nExiting application. Goodbye!")
                sys.exit(0)
            else:
                print("\nInvalid option. Please try again.")
                
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    app = hERGBlockerApp()
    app.run()