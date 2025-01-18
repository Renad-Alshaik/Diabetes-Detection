# Diabetes Classification Using XGBoots
This project demonstrates the use of various machine learning models to predict diabetes based on a dataset. The dataset contains both numerical and categorical features, and the pipeline focuses on preprocessing, model training, and evaluation.

Key Steps:
Data Preprocessing:
The dataset is loaded and the distribution of each feature is visualized using histograms and count plots.
Missing values in the smoking_history column are handled by removing rows with missing data.
Categorical variables, such as gender and smoking_history, are one-hot encoded.

Exploratory Data Analysis (EDA):
Basic information about the dataset (shape, data types, and statistics) is displayed.
A correlation matrix is computed to examine relationships between features and the target variable (diabetes), which is visualized using a heatmap.

Feature Engineering:
The dataset is split into features (X) and the target variable (y), where diabetes is the target.
A stratified train-test split ensures that the target class distribution is maintained.
SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the class distribution in the training data.
Features are scaled using StandardScaler to ensure all features have the same scale.

Model Training and Evaluation:
An initial XGBoost model is trained on the resampled data and evaluated on the test set.
The performance of the initial model is assessed using accuracy, confusion matrix, ROC curve, and classification report.
Hyperparameter tuning is performed using GridSearchCV to optimize the model's performance.
Comparison is made between XGBoost, KNN, and Decision Tree models based on accuracy, precision, recall, and F1-score.
ROC curves are plotted for all models to compare their performance visually.

Results:
After tuning, the XGBoost model shows improved performance in terms of precision, recall, and F1-score.
Visualizations like confusion matrices, ROC curves, and bar plots of model comparisons are presented to aid in interpreting the results.
This project uses libraries such as pandas, numpy, scikit-learn, xgboost, seaborn, and matplotlib for data processing, modeling, and visualization.

Technologies Used:
Python
scikit-learn
XGBoost
SMOTE (from imbalanced-learn)
pandas, numpy
Matplotlib, Seaborn

The Dataset used was take from kaggle
