import time
import pandas as pd
import logging
from pathlib import Path
from ersilia import ErsiliaModel 

class Featurizer:
    def __init__(self, model_id, dataset_dir="../data/", log_dir="../logs/"):
        self.model_id = model_id
        self.dataset_dir = Path(dataset_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def featurize(self, input_file):
        """
        Featurizes the input CSV using the specified Ersilia model.

        Args:
            input_file (str): The name of the input file (without extension).

        Returns:
            str: Path to the output featurized CSV file.
        """
        try:
            t1 = time.time()
            
            dataset_path = self.dataset_dir
            input_file_path = dataset_path / f"{input_file}.csv"
            output_file_path = dataset_path / f"{input_file}_{self.model_id}_featurized.csv"

            if not input_file_path.exists():
                self.logger.error(f"Input file not found: {input_file_path}")
                return None

            self.logger.info(f"Loading Ersilia model: {self.model_id}")
            model = ErsiliaModel(model = self.model_id)
            model.serve()
            model.run(input=str(input_file_path), output=str(output_file_path))

            featurized_df = pd.read_csv(output_file_path)

            path = self.dataset_dir / f'{input_file}.csv'
            if not path.exists():
                self.logger.warning(f"hERG dataset not found: {path}")
            else:
                df = pd.read_csv(path)
                featurized_df = featurized_df.merge(df[["Drug", "Y"]], left_on="input", right_on="Drug", how="inner")
                featurized_df.drop(columns=["Drug"], inplace=True)
                featurized_df.to_csv(output_file_path, index=False)

            time_taken = time.time() - t1
            self.logger.info(f"Featurization completed in {time_taken:.2f}s for {input_file_path}")
            
            return str(output_file_path)

        except Exception as e:
            self.logger.error(f"Error during featurization for model {self.model_id}: {e}", exc_info=True)
            return None