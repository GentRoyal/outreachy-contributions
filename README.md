# Toxicity Prediction Tasks (hERG Blockers)
This repository contain all the codes, dataset, scripts and figures used for the prediction of hERG blockers. The human either-a-go-go-related gene is a gene code for a potassium ion channel which is important for regulating heartbeat. We human take medications everytime for one or many purposes and we expect these medications to make us feel better. However, we can block our herg gene without knowing if we take drugs that is capable of blocking herg gene. Blocking this gene would cause a serious problem in the human system, so it is important that a model exist to help predict if a drug can block the herg gene and that is the essence of this project

## Overview
Specifically, I will be focus on creating a model that predicts whether a drug blocks the human ether-à-go-go related gene (hERG). The project is a **classification problem** and according to the project's dataset, each drug is represented by a drug id, a SMILES string and labeled as either blocking (1) or not blocking (0). The dataset includes 648 drugs and can be found under the [Toxicity Prediction Task](https://tdcommons.ai/single_pred_tasks/tox) in [TDCommons](https://tdcommons.ai/)

### Why this Project?
I must say that this wasn't the project I selected initially, I selected [DrugResponse - GDSC1](https://tdcommons.ai/multi_pred_tasks/drugres) but was informed it might get complicated especially when it gets to the featuriser stage and was advised to make a switch. So, I decided to carefully go through [TDCommons](https://tdcommons.ai/) again and I found hERG Blockers project under [Toxicity Prediction Task](https://tdcommons.ai/single_pred_tasks/tox). I am working on CareWomb, an AI/ML app that monitors maternal and fetal health, including heartbeats, to support safer pregnancies in remote areas. While the hERG blocker project and my CareWomb project are not directly related, I think there might be possiblities of conencting the two. The hERG blocker project is predicting whether drugs can block the hERG, and since the [hERG contributes to the electric activity of the heart](https://en.wikipedia.org/wiki#:~:text=This%20ion%20channel%20(sometimes%20simply%20denoted%20as%20%27hERG%27)%20is%20best%20known%20for%20its%20contribution%20to%20the%20electrical%20activity%20of%20the%20heart) and CareWomb tracks heatbeats, I think I can explore this possibility of linking the two projects and see where it leads.

## Task 1: Downloading the Dataset
### Characteristics
| Field         | Description                          |
|--------------|----------------------------------|
| **Drug_ID**   | Unique compound identifier        |
| **Drug**      | Represented by SMILES string (molecular structure) |
| **Y**         | Binary classification, 0: Not Blocker, 1: Blocker       |
| **Metric** | pIC50 (no unit) |

There were 22 missing values in the `Drug_ID` column and I filled the missing values by combining the PUBCHEMPY and RDKIT libraries. I started by extracting the molecule formula from the SMILES. This was then used to find the chemical formula and by extension, the drug name.

All through the project, I use a logger to log the process and the following is an extract
```bash
2025-03-26 09:32:58,292 - INFO - Processing 19 rows with missing Drug_IDs in ../data/train.csv...
2025-03-26 09:33:33,810 - INFO - Processed: {'SMILES': 'NC(=O)C[C@@H](N)c1nn[nH]n1', 'Formula': 'C4H8N6O', 'Name': 'MELAMINE FORMALDEHYDE', 'Drug_ID': 'MELAMINE FORMALDEHYDE'}
...
```

### Label Distribution
I looked at the distribution of labels and I found that the dataset contains a highly imbalanced distribution of labels, with more hERG blockers (451) than non-blockers (204). 
We need to address this imbalance because it is important before training and evaluating our model.
The label distribution can be found [here](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/label_distribution.png)

### SMILES Length Distribution
Also, when we look at the length of these SMILES strings, we see that it varies significantly. This is plotted [here](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/smiles_length.png), where most molecules have SMILES lengths around 30 to 70 characters.

### Sample Drug Structures
To better understand the dataset, I visualized some molecular structures of drugs in the dataset. This visualization can help in analyzing the chemical properties of hERG blockers and non-blockers.
The sample drug structures can be found [here](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/sample_molecules.png)

### Repository Ogranization
```bash
.
├── data/               # Raw and processed dataset files
│   └── figures/        # Figures from EDA
├── hERG.csv
├── hERG.tab
├── test.csv
├── train.csv
├── validation.csv
├── train_{featurizer_code}_featurised.csv
├── validation_{featurizer_code}_featurised.csv
├── test_{featurizer_code}_featurised.csv
├── placeholder.csv # Placeholder for featurized SMILE during predictions
├── new_compounds.csv # Contains the new compounds used to evaluate the models' performance
├
├── models/             # Saved machine learning models (pickle file)
├── best_eos2gw4_model.pkl 
├── best_eos5guo_model.pkl 
├── notebooks/          # Jupyter notebooks for analysis
│   └── TDC - Toxicity Prediction Task.ipynb
├── scripts/            # Python utility scripts
│   ├── main.py
│   ├── load_dataset.py
│   ├── clean_transform_data.py
│   ├── exploratory_analysis.py
│   ├── feature_engineering.py
│   ├── train_evaluate_model.py
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```
### File Naming Convention
To maintain consistency and clarity for featurised files, I used the convention [filename_{featurizer_code}_featurised.csv] for naming featurised datasets.

Take for instance, featurising train.csv file using ErG 2D featuriser (eos5gu0) would give a  featurised file with the name train_eos5gu0_featurised.csv
I did this to make sure that each dataset can be identified easily and that the file names are uniform across all splits of the dataset.

### SMILES Format

| Aspect  | Details |
|---------|---------|
| **Format**  | Text Notation of Molecules |
| **Length**  | 52 characters (Mean) |
| **Example** | `O=C1C=CC[C@@H]2[C@H]3CCCN4CCC[C@H](CN12)[C@H]34` (Sophocarpine) |

### Class Distribution

| Split      | Class 0 | Class 1 |
|-----------|----------------|----------------|
| **Overall**   | 69% | 31% |
| **Training**  | 69% | 31% |
| **Validation** | 62% | 38% |
| **Test**      | 73% | 27% |

### Prerequisites
- Python 3.12.9 or later
- Conda
- Jupyter Lab (optional)
- WSL (optional)
- Docker

### Loading Instructions
#### 1. Create Conda Environment
```bash
# Create a new conda environment
conda create --name ersilia python=3.12 -y
conda activate ersilia
```

#### 2. Install Dependencies
```bash
# Install required libraries
pip install -r requirements.txt
```

#### 3. Clone the Repository
```bash
git clone https://github.com/GentRoyal/outreachy-contribution.git
cd outreachy-contribution
```

#### How to Run
```bash
cd scripts
python main.py
```
- Follow Prompt to interact
```
==================================================
             hERG BLOCKER PREDICTION SYSTEM
==================================================
 NOTE: All input files should be placed inside the 'data' folder.
==================================================
 INSTRUCTIONS:
 You can featurize compounds using any of the following:
  - eos5gu0  ->  ERG 2D Featurizer
  - eos2gw4  ->  Ersilia Compound Embeddings
 The best-performing model for each featurizer can be used to generate
 predictions or assess performance on unseen data.
==================================================
 1. Download Dataset
 2. Featurize Dataset
 3. Fill in Missing Drug Names
 4. Generate Exploratory Data Analysis
 5. Assess Performance on Unseen Data
 6. Predict hERG Blocker Status
 7. Exit
==================================================
 Please select an option (1-7):
 ```
- Note: For WSL users experiencing Matplotlib crashes, set Qt platform:
```bash
echo 'export QT_QPA_PLATFORM=offscreen' >> ~/.bashrc
source ~/.bashrc
```
Although this project runs the hERG dataset, it is capable of doing the same operations on the following datasets {'LD50_Zhu', 'ClinTox', 'Carcinogens_Lagunin', 'Skin Reaction', 'AMES', 'hERG', 'hERG_Karim', 'DILI'}, if I lift the restriction on inputs.

**Option 1: Download Dataset: ** Selecting this option downloads all hERG related datasets (hERG.csv, hERG.tab, train.csv, validation.csv, test.csv)

**Option 2: Featurize Dataset: ** Selecting this option lets you featurise a dataset. When this option is selected, the available featurisers (i.e eos5gu0 and eos2gw4) are displayed and you also get a prompt to enter the name of the file to featurise.

**Option 3: Fill in Missing Drug Names: ** Selecting this option looks through the Drug_ID column of a dataset for missing Drug IDs (i.e. the drug names) and fill these missing ID using the PUBCHEMPY and RDKIT libraries

**Option 4. Generate Exploratory Data Analysis: ** Selecting this option performs a bried exploratory data analysis on the hERG dataset. It creates visuals (The distribution of SMILE lengths and the class distribution) and also prints the number of drugs in each class.

**Option 5. Assess Performance on Unseen Data: ** This option lets you run an already trained model on a new data to evaluate how the model performs on an unseen data. These models are the best models I got from each of the featuriser I applied. So, you get the option to select one of two and apply it to the new dataset. In the end, metrics like NPV, Specificity, ROC-AUC and accuracy score are displayed.

**Option 6. Predict hERG Blocker Status:**

**Option 7. Exit the Application:** This option ends the running of the application.

This option lets you enter a SMILE, select a model and generate prediction if the SMILE is a hERG blocker or not. The probability is also displayed.

## Task 2: Featurisation
The choice of Featuriser for this project are based on two criterions
1. The featuriser should be related to toxicity because it directly address the main goal of the project
2. The featuriser be related to drugs as this would be related to the dataset.
3. The featuriser should address bioactivities of molecules.

I started by checking the [Cardiotoxicity Classifier](https://github.com/ersilia-os/eos1pu1) but I later found out it's actually not a featuriser but a classification model that predicts if a drug is toxic or not.

Then I checked [DrugTax: Drug taxonomy](https://github.com/ersilia-os/eos24ci) which takes SMILES inputs and classifies the molecules according to their taxonomy (organic or inorganic), but this doesn't really have a direct relation with hERG blocking This Featuriser is easy to uses and it also uses a binary classification. 
In the end I got a vector of 163 features including the taxonomy classification.

Lastly, I checked the [ErG 2D Descriptors](https://github.com/ersilia-os/eos5guo) featuriser.
This featuriser focuses on bioactivity, and it also captures pharmacophoric properties, size and shape of molecules. Since, we're dealing with a drug dataset, I think it's an ideal alternative to check.

```bash
Performing Featurisation...
2025-03-27 08:39:00,647 - INFO - Featurization completed in 5.40s for ../data/train.csv
2025-03-27 08:39:00,649 - INFO - Loading Ersilia model: eos5guo
2025-03-27 08:39:15,387 - INFO - Featurization completed in 14.74s for ../data/test.csv
2025-03-27 08:39:15,388 - INFO - Loading Ersilia model: eos5guo
2025-03-27 08:39:33,235 - INFO - Featurization completed in 17.85s for ../data/validation.csv
```

## Task 3: Build an ML Model
### Addressing Class Imbalance
The train set is imbalanced (314 class 1 and 144 class 0), I addressed it by applying these three techniques on my train set.
- imblearn.oversampler
- SMOTE
- Hybrid sampler (SMOTE-EEN) 

So, there are four train sets in total i.e. the original train set and the three sampled train sets.
[The distributions of the sampling are attached here](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/train_sets.png)

- The imblearn.oversampler and SMOTE distribution appear to be the same but their methods of Oversampling are different (imblearn.oversampler samples the smaller class randomly while SMOTE sampling uses synthetic samples)
- The hybrid sampling combined oversampling and undersampling to retain a more representative dataset but it does not fully equalize the classes like SMOTE or random oversampling.

### Further Preprocessing  
- I dropped single-valued columns from the featurized sets.  
- I scaled train, validation, and test features with `StandardScaler`.  
- Then I applied the above sampling techniques to address class imbalance.

### Algorithms Used
- XGBoostClassifier
- RandomForestClassifier
  
### Algorithm Configurations
Instead of training each model separately, I created a function that accepts different configurations and runs the models accordingly. 
This allowed me to compare how different settings affect performance. 

The configurations I used are:
- Basic setup: No class weighting, no stratified K-Fold, and no hyperparameter tuning.
- Class weighting only: Adjusts for class imbalance by giving more weight to the minority class.
- Stratified K-Fold only 
- GridSearchCV only
- Class weighting + Stratified K-Fold
- Class weighting + GridSearchCV

It is worth noting that the dataset was splitted using scaffold method of split. 
This is because I think we stand a chance of making sure that similar molecules don’t mix between our training, validation and test sets if we use scaffold split. By that we are truly testing our model on new data instead of a random split.

### Evaluation Metrics

| **Metric**  | **Description**  | **Formula** |
|------------|----------------|------------|
| **Accuracy** | Proportion of correctly classified instances | (TP + TN) / (TP + TN + FP + FN) |
| **F1-Score**  | Harmonic mean of precision and recall | 2 * (Precision * Recall) / (Precision + Recall) |
| **ROC-AUC**  | Measures the ability to distinguish between classes | Area under the Receiver Operating Characteristic (ROC) curve |
| **Specificity**  | Proportion of actual negatives correctly predicted (True Negative Rate) | TN / (TN + FP) |
| **Negative Predictive Value**  | Proportion of true negatives among predicted negatives | TN / (TN + FN) |

### Training and Evaluating The Model
**For DrugTax featurized Dataset**
- After removing the single valued columns, the dataset of a total of 163 features remained just 40 features.
- Both algorithms performed poorly on the sampled and unsampled variations of this dataset.
- The metrics decreased significantly from the training set to the validation set and then the training set. It does not only show that the model overfits, it also show that the model does not generalize during training.

| Metric          | Train  | Validation   | Test    |
|----------------|---------|--------------|---------|
| Accuracy       | 92.78%  | 82.09%       | 80.71%  |
| Precision      | 88.71%  | 89.47%       | 82.61%  |
| Recall         | 95.93%  | 80.95%       | 93.14%  |
| F1-Score       | 92.18%  | 85.00%       | 87.56%  |
| ROC AUC        | 97.85%  | 87.29%       | 78.46%  |

**For ERG 2D featurized Dataset**
- The ERG 2D Featurised Dataset showed improvement over the DrugTax featurizer.
- The model still generalized poorly especially with a decrement from train to validation metrics

| Metric          | Train  | Validation   | Test    |
|----------------|---------|--------------|---------|
| Accuracy       | 84.85%  | 76.12%       | 77.14%  |
| Precision      | 82.63%  | 74.07%       | 76.52%  |
| Recall         | 98.74%  | 95.24%       | 99.02%  |
| F1-Score       | 89.97%  | 83.33%       | 86.32%  |
| ROC AUC        | 95.40%  | 79.57%       | 82.24%  |

I created a [confusion matrix](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Confusion%20Matrix%20-%20ErG%202D%20Descriptors.png) to analyze label classifications, a [precision-recall curve](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Precision-Recall%20Curve%20-%20ErG%202D%20Descriptors.png) which shows the trade-off between precision and recall. But the precision-recall curve is not my focus because this graph is more relevant when we need to identify more positive cases (blockers).
In this case, we need to correctly classify more negative classes (non blockers) due to the class imbalance, so I used a [ROC-AUC curve](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/ROC%20Curve%20-%20ErG%202D%20Descriptors.png) to assess classification quality, combined with the confusion matrix.
The ROC-AUC curve shows the model is 82.24% confident in its prediction but we can't look away from the sensitivity and negative class prediction rate from the confusion matrix.

The 87.50% negative predictive value means the model gets 87.50% of the no blockage classifications correctly which is quite good but the 18.42% specificity means the model really struggle to classify negative class (No Blockage), which is very low.
The 18.42% means that the remaining 81.58% were drugs that can block the hERG but misclassified as non-blockers. This percentage is too high; I would say it is unacceptable and it is too risky to deploy such model even though the ROC-AUC curve shows that the model is 82.24% confident in its prediction.

So, we need to find a way to improve the model's specificity or train another model.
I also included a [top 10 feature importance visual](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Top%2010%20Feature%20Importances%20-%20ErG%202D%20Descriptors.png) to highlight key predictors.

I need to say at this point that the featurisers did not return descriptive names as the column names except for a few, so it's difficult to say what each feature represents.
If we can get the descriptive feature names, the feature importance visual would be more interpretable.

### Comparison with other Featuriser
Another featuriser I tried is the Ersilia Compound Embeddings. According to the description of this featuriser, it generates bioactivity-aware chemical embeddings, combining physicochemical and bioactivity information; So I thought of giving it a try.
At first, I got this as the best metric 

| Metric          | Train  | Validation   | Test    |
|----------------|---------|--------------|---------|
| Accuracy       | 100.0%        | 77.61%            | 76.43%             |
| Precision        | 100.0%              | 84.62%            | 85.57%             |
| Recall           | 100.0%              | 78.57%            | 81.37%             |
| F1_Score        | 100.0%              | 81.48%           | 83.42%             |
| ROC_AUC         | 100.0%              | 82.67%            | 82.56%             |

The model clearly overfits the train set because of its perfect rating on the train set but overall, the model generalize moderately well and it struggles with unseen data because some of the metrics keeps decreasing.
Similarly, I the same figures I created for the ErG 2D model; a [confusion matrix](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Confusion%20Matrix%20-%20Ersilia%20Compound%20Embeddings.png), a [precision-recall curve](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Precision-Recall%20Curve%20-%20Ersilia%20Compound%20Embeddings.png), a [ROC-AUC curve](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/ROC%20Curve%20-%20Ersilia%20Compound%20Embeddings.png) and a [top 10 feature importance visual](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/Top%2010%20Feature%20Importances%20-%20Ersilia%20Compound%20Embeddings.png) to highlight key predictors.

To decide which model is better, I used a trade-off of which poses more risk "misclassifying a drug as hERG blocker or misclassifying a drug as not" i.e. False Positive vs False Negative

**False Negative (misclassifying a hERG blocker as not a blocker): ** To the best of my knowledge, this is more dangerous. A drug that is a hERG blocker could cause serious heart issues, and misclassifying it as not can cause serious health issues.

**False Positive (misclassifying a non-hERG blocker as a blocker): ** This is a serious error too but I think the error is less risky. The worse that could happen here could be delaying the approval of the drug before it is considered safe, but it doesn’t pose a direct health risk.

Now using this as a benchmark, I can now compare the two models:
ErG 2D Description model has high NPV (87.50%), meaning that the model gets 87.50% of the no blockage classifications correctly which is quite good, but it has low specificity (18.42%), meaning that the model only gets 18.42% of actual non-hERG blockers. 

The Ersilia Compound Embeddings model on the other hand has lower NPV (55.81%) but higher specificity (63.16%), meaning it's better at correctly identifying drugs that are not blockers compared.

Using my trade-off, I gave priority to minimizing the risk of missing a dangerous hERG blocker, so I chose the Ersilia Compound Embeddings model. Even though its NPV is lower, its higher specificity makes it better at avoiding the false negative that could lead to serious health risks.

I expanded my GridSearchCV hyperparemeters, and I got a better metric in terms of the [ROC AUC](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/ersilia_embedding_roc_auc_second.png).
| Metric          | Train  | Validation   | Test    |
|------------------|------------------|--------------------|--------------------|
| Accuracy         | 97.96%              | 82.14%            | 74.62%             |
| Precision        | 97.21%              | 81.81%            | 73.58%             |
| Recall           | 98.74%              | 97.05%            | 92.85%             |
| F1_Score        | 97.97%              | 88.79%           | 82.11%             |
| ROC_AUC         | 99.82%              | 78.36%            | 89.43%             |

But this came at a price, the specificity according to the [confusion matrix](https://github.com/GentRoyal/outreachy-contributions/blob/main/data/figures/ersilia_embedding_confusion_matrix_second.png) reduced to 44.00% and the NPV increased to 78.57%.
A reduced specificity of 44.00% means that the model cannot correctly classify negative classes as well as the previous model with a better specificity of 63.16%

### So, which model is the Best and Why?
In terms of NPV/Specificity trade-off, the first Ersilia Compound Embeddings model performed better than all other models. It had a specificity of 63.16% and I'd give preference to this model than others because the class distribution is imbalance and a model that works well with the negative class should be given preference.
While in terms of ROC-AUC, the second Ersilia Compound Embeddings model performed better than all other models with a ROC-AUC of 89.43% and it even performed better than current [leaderboard](https://tdcommons.ai/benchmark/admet_group/hERG)

I think the ROC-AUC shouldn't be used solely as the metric for this project because it does not give the complete picture of the models' classification quality. 
A model can have a good ROC-AUC but fail to capture the negative classes. Combining the ROC-AUC and the NPV metrics would give us a better benchmark to select a good model.

My conclusion is that the model that gave me the best specificity (i.e. the first Ersilia Compound Embeddings model) is the best model because it performed well with the negative class and it also has a good ROC-AUC of 82.56%.

### Performance on unseen data
I did got additional data from [hERG blocker/non-blocker datasets](https://weilab.math.msu.edu/DataLibrary/2D/Downloads/hERG-classification.zip), it's a zip file that contains 7 other zipped files, all of which can be used to validate my model.
I picked one of them and extracted 43 compounds that weren't part of my original dataset of which, 18 are hERG blockers (class 1) and 25 are non-blockers (class 0).

I ran the Ersilia Compound Embeddings model on the extracted compounds and this was its performance.

| **Metric**                          |    **Value** |
|--------------------------|--------------------|
| Accuracy                        | 51.35% |
| Specificity                     | 100%        |
| Negative Predictive Value (NPV) | 51.35% |
| ROC-AUC                         | 80.41% |

For the best ERG 2D model, I got this performance metric:
| **Metric**                          |    **Value** |
|--------------------------|--------------------|
| Accuracy                        | 59.46% |
| Specificity                     | 100%        |
| Negative Predictive Value (NPV) | 55.88% |
| ROC-AUC                         | 75.15% |

Since both models had 100% specificity (meaning all the negative classes they predicted were accurate), we can rely on another metric to compare their performances. 
In this case, we use the ROC-AUC. 
The Ersilia Compound Embeddings model had a higher ROC-AUC, ant this supports my choice of it as the better model.