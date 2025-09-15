# Machine Learning for Precision Oncology: Predicting Personalized Anti-Cancer Treatments for 200,000+ Breast Cancer Patients

In this project, we built a machine learning pipeline to match breast cancer patients with personalized anti-cancer therapeutics and predicted their potential treatment responses.

Our dataset comprised 204,026 breast cancer patients and incorporated 86 clinical, experimental, and drug-related features aggregated from multiple heterogeneous sources. The raw data required extensive cleaning, normalization, and integration to ensure quality and comparability across studies. This process included resolving missing values, unifying feature definitions, and harmonizing measurement scales to form a single, consistent analytical dataset.

In addition, We utilized machine learning and advanced data processing techniques to analyze complex, multi-source datasets consisting 86 dimensions. This included dimensionality reduction methods (PCA, UMAP), feature attraction of drug characterization metrics (IC50 calculations, PK/PD measures), and a suite of machine learning algorithms — including Gaussian Naive Bayes, Nearest Neighbors, feed-forward neural networks, binary classifiers, and k-fold cross-validation — to build predictive models for personalized anti-cancer therapeutics and patient treatment response.

### The flow of analysis and notebooks in this directory should be ran in the following order. Each notebook builds on top of the other:
1) main.ipynb --> Initial notebook used to combine multiple disjoint datasets containing patient demographic information. 
2) austin_race_analysis.ipynb --> imputes missing racial information with naive bayes.
3) Lucas_Cell_Line_alignment.ipynb --> Utilizes processed data set with no missing racial information to map cell line to patient IDs based on shared patient demographics (race, age, and t-stage of breast cancer).
4) joyce_SmallMoleculesEDA_preprocessing_Submission.ipynb, joyce_IC50_Determination_Submission.ipynb, joyce_IC50_Prediction_Baseline_Submission.ipynb, joyce_IC50_Prediction_exclude_cells_Submission.ipynb, joyce_IC50_Prediction_with_cells_Submission.ipynb, CHEM277B_functions.py --> Utilizes dataset generated from Lucas_Cell_Line_alignment, as well as processed IC50 determinations, and Small molecules EDA to map chemotherapies to patients and generate models for IC50 predictions using simple feed forward neural network.
5) Sam_failed_life_models.ipynb, Sam_NN --> Initial and final models for patient survival prediction based on their treatment plan (chemotherapy vs. radiotherapy).
6) hbling_UMAP.ipynb --> EDA of UMAP projects on raw datasets and final datasets generated from models/notebooks mentioned above. 