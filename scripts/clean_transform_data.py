import logging
import time
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv

    def process_csv(self):
        """
        Reads a CSV file, fills missing Drug_IDs, and saves the processed data.
        """
        try:
            df = pd.read_csv(self.input_csv)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        if "Drug" not in df.columns or "Drug_ID" not in df.columns:
            logger.error("CSV must contain 'Drug' and 'Drug_ID' columns.")
            return

        missing_ids = df[df["Drug_ID"].isna()]
        if missing_ids.empty:
            logger.info("No missing Drug_IDs found. No processing needed.")
            return

        logger.info(f"Processing {len(missing_ids)} rows with missing Drug_IDs in {self.input_csv}...")

        for index, row in missing_ids.iterrows():
            smile = row["Drug"]
            mol = Chem.MolFromSmiles(smile)

            if mol:
                formula = rdMolDescriptors.CalcMolFormula(mol)
                try:
                    compounds = pcp.get_compounds(formula, 'formula')
                    if compounds:
                        name = compounds[0].synonyms[0] if compounds[0].synonyms else compounds[0].iupac_name
                        drug_id = name
                    else:
                        name = formula  # Fallback to formula
                        drug_id = formula

                    df.at[index, "Drug_ID"] = drug_id  

                    log_entry = {"SMILES": smile, "Formula": formula, "Name": name, "Drug_ID": drug_id}
                    logger.info(f"Processed: {log_entry}")
                except Exception as e:
                    logger.error(f"Error fetching data for {formula}: {e}")
                    df.at[index, "Drug_ID"] = formula  # Use formula as fallback

                    log_entry = {"SMILES": smile, "Formula": formula, "Error": str(e)}
                    logger.info(f"Processed: {log_entry}")

                # Pause to avoid API rate limits
                time.sleep(5)
            else:
                logger.error(f"Invalid SMILES: {smile}")

        df.to_csv(self.output_csv, index=False)
        logger.info(f"Processed data saved to {self.output_csv}")
