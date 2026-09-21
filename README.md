Unit 1 Tabular Data

Notebook 1- EDA

Project Overview
Before jumping into building prediction models, I put together a thorough Exploratory Data Analysis (EDA) on this dataset to get a clear picture of what we’re actually working with. The goal was simple: map out the data’s overall structure, spot interesting patterns and relationships with price, and uncover every data quality issue that needs fixing before model development.
What We Built & Analyzed
Data Profiling & Quality Checks
Checked dataset dimensions and data types across every column.
Tallied up missing values to see where the biggest data gaps were.
Flagged columns that didn't add predictive value so we can drop them early.
Looked for hidden errors like typos, malformed strings, and incorrect data types, creating a clear list of things to fix during data preprocessing.
Looking at Individual Features (Univariate Analysis)
Used histograms and summary stats to look at numerical columns, keeping an eye out for extreme outliers and skewed data.
Checked categorical features to see their distributions and flagged high-cardinality columns with too many unique values.
Finding Relationships & Drivers (Bivariate Analysis)
Analyzed how different features relate to price to figure out which ones carry the most predictive weight.
Ran correlation checks between features to catch multicollinearity issues early on.
Documenting Practical Insights
Instead of just dropping plots into the notebook, I added brief explanations under key charts and tables to translate the visuals into practical findings

Notebook 2 - Preprocessing

Project Overview
With the initial data exploration out of the way, this notebook focuses on turning the raw dataset into a clean, leakage-free, model-ready format. The main objective here was to build a rigorous preprocessing pipeline that handles everything from messy data types and missing values to advanced feature engineering and target scaling while strictly protecting the model’s integrity and preserving essential metadata for downstream fairness auditing.
What We Built & Executed
Data Cleaning & Structure Normalization
Loaded the raw data from scratch to ensure a clean, reproducible pipeline independent of prior environment states.
Dropped irrelevant columns and unsalvageable rows, documenting a clear domain justification for every removal.
Resolved mixed data types across tricky columns to establish consistent, unified data types.
Outlier Strategy & Controlled Data Preservation
Cleaned up severe outliers across key numerical features (year, price, and mileage).
Enforced programmatic checks to verify that combined row drops from null handling and outlier trimming stayed well under the 50% threshold to preserve dataset utility.
Domain-Informed Imputation & Feature Engineering
Implemented tailored imputation strategies grounded in domain logic rather than relying solely on generic median or mode fills.
Extracted high value features from complex inputs (such as parsing major_options and timestamp details from listed_date).
Integrated all required community and instructor features, ensuring zero target leakage by treating price as strictly off-limits during feature derivation.
Encoding, Metadata Retention & Scaling
Applied label encoding to binary features and used frequency capped along with one-hot encoding for the top 20 categories with reference dropping 
Preserved original categorical text columns as metadata alongside encoded dummies to support the fairness and bias audit in Notebook 5
Standardized all non metadata features using StandardScaler while keeping the target variable price in its original scale.
Dataset Splitting & Export
Executed a deterministic 80/20 train/test split (with random_state=42).
Exported processed_train.csv and processed_test.csv for model training in Notebook 3.
Notebook 3 - Modeling

Notebook 3: Regression Modeling, Cross-Validation & Architecture Comparison
Project Overview
With our preprocessed training and test sets established, our team focused on evaluating multiple regression algorithms to predict car prices. We benchmarked classical machine learning models against deep neural network architectures using rigorous 5 fold cross validation and systematically tuned hyperparameters to evaluate the models performance and identify our top performing estimators.
What We Built & Executed
Data Preparation & Leakage Defense:
Imported our deterministic train/test splits (processed_train.csv and processed_test.csv).
Separated our regression target (price) and excluded all categorical metadata attributes from our feature matrices to ensure clean, leak-free training across both splits.
Scikit Learn Model Exploration & Hyperparameter Tuning:
Evaluated at least three distinct algorithmic families, including Linear Regression, Random Forest Regressor, and additional ensemble models.
Conducted systematic hyperparameter searches using 5-fold GridSearchCV across custom parameter grids.
Retrained winning estimators on the entire training dataset via refit=True to extract final optimized models.
Keras Deep Learning Architecture Search:
Designed and benchmarked three distinct neural network architectures varying in depth, layer density, and activation strategies.
Implemented a manual 5 fold cross validation pipeline across architectures to assess cross fold performance stability.
Select the top performing network configuration based on average cross validation metrics and refit the final architecture on the complete training set.
Performance Benchmarking & Evaluation:
Evaluated every refit model on the holdout test set (processed_test.csv).
Logged key evaluation metrics across training and test phases: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Coefficient of Determination (R2), and total refit execution time.
Compiled all results into a unified benchmark table to directly compare trade-offs in accuracy, variance, and computational cost.
Model Serialization & Metric Exports:
Exported the final consolidated performance summary as model_metrics.csv.
Saved all best performing Scikit Learn estimators files and exported the winning Keras model
Notebook 4: Feature Selection
Project Overview
After building our initial models, our team wanted to see if we could simplify our dataset without losing accuracy. We used Random Forest feature ranking and Variance Inflation Factor (VIF) filtering, to cut out weak or repetitive features. Then, we retrained all of our models on this smaller feature set to see how much speed and performance we gained (or lost).
What We Did
Setup & Baseline Checks
Loaded our train and test splits (processed_train.csv and processed_test.csv) to set up our features and target (price).
Brought in our original performance results from Notebook 3 (model_metrics.csv) as a benchmark.
Cutting Down the Feature Count
Step 1 (Top 100 Features): Trained a Random Forest model to rank features by importance and kept only the top 100.
Step 2 (Removing Redundancy): Used VIF to spot highly correlated features and dropped any with a VIF score over 10.
Retraining Our Models
Retrained all of our Scikit Learn models on the smaller feature set using 5 fold cross validation to find the best settings.
Retested our Keras neural network architectures on the trimmed features, cross-validated their performance, and refit the best one on the full training set.
Comparing Before vs. After 
Calculated the new metrics (MAE, MAPE, R2) on the test set and measured training times.
Created a side by side comparison table (model_metrics_selected.csv) to track performance shifts.
Discussed the trade-offs: which models got faster, which maintained accuracy, and which suffered from having fewer features.
Saving Our Work
Saved the trimmed datasets as selected_train.csv and selected_test.csv, making sure to keep price and raw metadata columns intact.
Exported the updated results to model_metrics_selected.csv.
Saved all retrained models to our models/ folder 
Notebook 5: Interpretability & Fairness
Project Overview
In our final notebook, our team opened up the "black box" of our top-performing model to understand how it makes predictions and check for potential algorithmic bias. We used SHAP to identify which features drive car price predictions, followed by a fairness to evaluate how model performance holds up across different metadata subgroups.
What We Did
Setup & Model Loading
Loaded our reduced test set (selected_test.csv) and brought in our saved Random Forest model (_selected.pkl) from the models/ directory.
SHAP Interpretability Analysis
Global Feature Impact: Generated a SHAP beeswarm plot on a random sample of test data to visualize which features most strongly push price predictions up or down across the board.
Case-by-Case Breakdown: Created SHAP force plots for individual predictions while analyzing at least two accurate predictions and two high error outliers to see what drove the model's successes and misses.
Sanity Check: Evaluated whether the features the model relies on most align with real-world car valuation dynamics.
Subgroup Bias & Fairness Audit
Defining Subgroups: Used the raw metadata text columns preserved since Notebook 2 to group predictions across at least three distinct categorical dimensions.
Disparity Metrics: Applied Fairlearn’s MetricFrame to calculate regression metrics (MAE, MAPE, R2) across each subgroup to measure performance gaps.
Visualizing Gaps: Built bar charts to highlight performance differences and variance across demographic/subgroup categories.
Discussion & Key Takeaways
Analyzed whether observed performance disparities stemmed from data imbalances, modeling limitations, or natural variations in the underlying market.
