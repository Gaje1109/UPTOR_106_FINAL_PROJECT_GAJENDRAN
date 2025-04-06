import warnings
warnings.simplefilter(action= "ignore")
warnings.filterwarnings("ignore")


# Import the packages
import numpy as np
import pandas as pd


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Algorithms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error


# Developer = R.Gajendran

# Import the CSV file
maternal_health_df= pd.read_csv("/MediPredict_AI_Health_Analyzer/CSVFiles/Maternal Health Risk Data Set.csv")

print("Dataframe loaded and opened: \n", maternal_health_df.head())
'''
print("Information about data set: \n",maternal_health_df.info())

print(" The dataset size: \n",maternal_health_df.shape)

print("Risk Level: ", maternal_health_df['RiskLevel'].value_counts())


print("\n Describe: \n ",maternal_health_df.describe().T)


# Plot stacked histograms for the feature variables
fig, axes= plt.subplots(nrows= 3, ncols= 2, figsize= (30,25 ))
risk_level_order= ["high risk", "mid risk","low risk"]

for ax, column in zip(axes.flatten(), maternal_health_df.columns):
    sns.histplot(data = maternal_health_df, x= column,kde= True, hue="RiskLevel", hue_order=risk_level_order, multiple= "stack",
                 palette= {"low risk":"green", "mid risk": "orange", "high risk":"red"}, element="bars", ax= ax)

    ax.set_title(f"{column}", fontsize= 25)

plt.tight_layout()
plt.savefig("MHP_Histogram.png")
plt.show()


# Plot Boxplots for the feature variables
fig, axes= plt.subplots(nrows= 3, ncols= 2, figsize= (30,25))
for ax, column in zip(axes.flatten(), maternal_health_df.columns):
    sns.boxplot(y =maternal_health_df[column],color = "#4682B4", ax = ax)

    ax.set_title(f"{column}", fontsize= 25)
plt.tight_layout()
plt.savefig("/MediPredict_AI_Health_Analyzer/Images/MHP_BoxPlot.png")
plt.show()

# CO RELATION

# Map risk levels to integer values
risk_mapping ={"low risk" :0 , "mid risk" :1 , "high risk" :2 }
maternal_health_df["RiskLevel"] = maternal_health_df["RiskLevel"].map(risk_mapping)
print("After Mapping: \n",maternal_health_df.info())

# Co relation graph
plt.figure(figsize= (22,20))
sns.heatmap(maternal_health_df.corr(), annot= True, cmap ="GnBu")
plt.title("Correlation heatmap of variables", fontsize= 16)
plt.savefig("/MediPredict_AI_Health_Analyzer/Images/MHP_Correlation.png")
plt.show()

'''
# Drop SystolicBP for model training
maternal_health_df = maternal_health_df.drop(['SystolicBP'], axis =1)

# Identify the outlier in Heart Rate
print("Outlier for Heart Rate :\n",maternal_health_df.HeartRate.sort_values().head())

# Remove the outlier in heart rate
maternal_health_df = maternal_health_df.drop(maternal_health_df.index[maternal_health_df.HeartRate == 7])

print("Data set after Outliers clean up: \n",maternal_health_df.info())

'''--------------------------------------------------------------------------------------------------------------------'''
# Model Building
columns_df = ['Age', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
scale_X= StandardScaler()
X= pd.DataFrame(scale_X.fit_transform(maternal_health_df.drop(["RiskLevel"],axis= 1)), columns=columns_df)
Y= maternal_health_df["RiskLevel"]

print("X: ",X)
print("y: ",Y)


# Train Test spilt
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify = Y)
print("X_train: \n",X_train.shape)
print("X_test: \n",X_test.shape)

print("Y_train: \n",y_train.shape)
print("Y_test: \n",y_test.shape)
print("-------------------------------------------------LOGISTIC REGRESSION-------------------------------------------------------------------")
'''-------------------------------------------LOGISTIC REGRESSION-----------------------------------------------------------------------'''

# Initializae Label encoder
label_encoder = LabelEncoder()

# convert categorical labels into numbers
y_train_encoded  = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)


# Baseline model of Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression_mod= logistic_regression.fit(X_train, y_train_encoded)
print(f"Baseline Logistic Regression : {round(logistic_regression_mod.score(X_test, y_test_encoded), 3)}")
pred_logistic_regression= logistic_regression_mod.predict(X_test)

# Cross Validate Logistic Regression Model
scores_Logistic = cross_val_score(logistic_regression_mod, X_test, y_test_encoded, cv= 3, scoring= "accuracy")
print(f"Scores(cross Validate) for Logistic Regression model: \n{scores_Logistic}")
print(f"Cross Val Means : {round(scores_Logistic.mean(), 3)}")
print(f"Cross Val Standard Deviation : {round(scores_Logistic.std(), 3)}")

# Test the new parameter
logistic_regression = LogisticRegression(C= 0.01, intercept_scaling= 1, max_iter= 100, solver= "liblinear", tol= 0.0001)
logistic_regression_mod= logistic_regression.fit(X_train, y_train_encoded)
pred_knn = logistic_regression_mod.predict(X_test)

mse_lr= mean_squared_error(y_test_encoded, pred_logistic_regression)
rmse_lr= np.sqrt(mse_lr)
scores_lr_train = logistic_regression_mod.score(X_train, y_train)
scores_lr_test = logistic_regression_mod.score(X_test, y_test_encoded)

print(f"Mean Square Error for LR = {round(mse_lr, 3)}")
print(f"Root Mean Square Error for LR = {round(rmse_lr, 3)}")
print(f"R^2(coefficient of determination) on training set= {round(scores_lr_train, 3)}")
print(f"R^2(coefficient of determination) on testing set= {round(scores_lr_test, 3)}")

'''
# Hyperparameter tuning
params_LR = {
    "tol": [0.0001, 0.0002, 0.0003],
    "C": [0.01, 0.1, 1, 10, 100],
    "intercept_scaling": [1, 2, 3, 4],
    "solver": ["liblinear", "lbfgs", "newton-cg"],
    "max_iter": [100, 200, 300],
}


GridSearchCV_LR= GridSearchCV(estimator= linear_model.LogisticRegression(),
                              param_grid= params_LR,
                              cv=3,
                              scoring= "accuracy",
                                        return_train_score= True)

GridSearchCV_LR.fit(X_train, y_train)

print(f"Best estimator for LR Model: \n {GridSearchCV_LR.best_estimator_}")
print(f"Best parameter values for LR Model: \n {GridSearchCV_LR.best_params_}")
print(f"Best score for LR Model: \n {GridSearchCV_LR.best_score_, 3}")


# Classification Report

print("Classification Report")
print(classification_report(y_test, pred_logistic_regression))
print("Confusion Matrix")
print(confusion_matrix(y_test, pred_logistic_regression))
'''
print("------------------------------------------------K-NEAREST NEIGHBOUR--------------------------------------------------------------------")
'''--------------------------------------------------------------------------------------------------------------------'''

# Baseline model of K-Nearest Neighbors
knn = KNeighborsClassifier()
knn_mod= knn.fit(X_train, y_train_encoded)
print(f"Baseline K-Nearest Neighbors: {round(knn_mod.score(X_test, y_test_encoded), 3)}")
pred_knn= knn_mod.predict(X_test)

# Cross Validate K Nearest Neighbor Model
scores_knn = cross_val_score(knn_mod, X_test, y_test_encoded, cv= 3, scoring= "accuracy")
print(f"Scores(cross Validate) for K nearest neighbors model: \n{scores_knn}")
print(f"Cross Val Means : {round(scores_knn.mean(), 3)}")
print(f"Cross Val Standard Deviation : {round(scores_knn.std(), 3)}")

'''
# Hyperparameter tuning
params_KNN = {
    "leaf_size": list(range(1,30)),
    "n_neighbors": list(range(1,21)),# Number of nearest neighbors
    "weights": ["uniform", "distance"],  # Weight function
    "p": [1, 2]  # Power parameter for Minkowski distance (1=Manhattan, 2=Euclidean)
}

GridSearchCV_knn = GridSearchCV(estimator= KNeighborsClassifier(),
                                param_grid= params_KNN,
                                cv= 3,
                                scoring= "accuracy",
                                return_train_score= True)
# Fit model with train data
GridSearchCV_knn.fit(X_train, y_train)

print(f"Best estimator for KNN Model: \n {GridSearchCV_knn.best_estimator_}")
print(f"Best parameter values for KNN Model: \n {GridSearchCV_knn.best_params_}")
print(f"Best score for KNN Model: \n {GridSearchCV_knn.best_score_, 3}")

'''
# Test the new parameter
knn = KNeighborsClassifier(leaf_size= 1, n_neighbors= 10, p= 2, weights= "distance")
knn_mod= knn.fit(X_train, y_train_encoded)
pred_knn = knn_mod.predict(X_test)

mse_knn= mean_squared_error(y_test_encoded, pred_knn)
rmse_knn= np.sqrt(mse_knn)
scores_knn_train = knn_mod.score(X_train, y_train_encoded)
scores_knn_test = knn_mod.score(X_test, y_test_encoded)

print(f"Mean Square Error for KNN = {round(mse_knn, 3)}")
print(f"Root Mean Square Error for KNN = {round(rmse_knn, 3)}")
print(f"R^2(coefficient of determination) on training set= {round(scores_knn_train, 3)}")
print(f"R^2(coefficient of determination) on testing set= {round(scores_knn_test, 3)}")


print("-----------------------------------------------RANDOM FOREST---------------------------------------------------------------------")
'''--------------------------------------------------------------------------------------------------------------------'''

# Baseline model of Random Forest
rf = RandomForestClassifier()
rf_mod = rf.fit(X_train, y_train_encoded)
print(f"Baseline Random Forest: {round(rf_mod.score(X_test, y_test_encoded), 3)}")
pred_rf = rf_mod.predict(X_test)

# Cross Validate Random Forest Model
scores_rf = cross_val_score(rf_mod, X_test, y_test_encoded, cv=3, scoring="accuracy")
print(f"Scores (Cross Validation) for Random Forest Model: \n{scores_rf}")
print(f"Cross Val Means: {round(scores_rf.mean(), 3)}")
print(f"Cross Val Standard Deviation: {round(scores_rf.std(), 3)}")
'''
# Hyperparameter tuning
params_RF = {
    "n_estimators": [50, 100, 200, 300],  # Number of trees
    "max_depth": [None, 10, 20, 30],  # Max depth of trees
    "min_samples_split": [2, 5, 10],  # Min samples required to split
    "min_samples_leaf": [1, 2, 4],  # Min samples per leaf
    "bootstrap": [True, False]  # Whether to use bootstrap samples
}

GridSearchCV_rf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params_RF,
    cv=3,
    scoring="accuracy",
    return_train_score=True
)

# Fit model with train data
GridSearchCV_rf.fit(X_train, y_train)

print(f"Best estimator for Random Forest Model:\n{GridSearchCV_rf.best_estimator_}")
print(f"Best parameter values for Random Forest Model:\n{GridSearchCV_rf.best_params_}")
print(f"Best score for Random Forest Model:\n{round(GridSearchCV_rf.best_score_, 3)}")
'''
# Test the new parameter
rf = RandomForestClassifier(n_estimators=100,   # Number of trees
    max_depth=10,       # Limit tree depth
    max_features="log2", # Reduce features per split
    min_samples_split=5, # Prevent overfitting
    random_state=42)
rf_mod = rf.fit(X_train, y_train_encoded)
pred_rf = rf_mod.predict(X_test)

mse_rf = mean_squared_error(y_test_encoded, pred_rf)
rmse_rf = np.sqrt(mse_rf)
scores_rf_train = rf_mod.score(X_train, y_train_encoded)
scores_rf_test = rf_mod.score(X_test, y_test_encoded)

print(f"Mean Square Error for RF = {round(mse_rf, 3)}")
print(f"Root Mean Square Error for RF = {round(rmse_rf, 3)}")
print(f"R^2 (coefficient of determination) on training set= {round(scores_rf_train, 3)}")
print(f"R^2 (coefficient of determination) on testing set= {round(scores_rf_test, 3)}")

print("---------------------------------------------GRADIENT BOOSTING-----------------------------------------------------------------------")
'''--------------------------------------------------------------------------------------------------------------------'''
# Baseline model of Gradient Boosting
gb = GradientBoostingClassifier()
gb_mod = gb.fit(X_train, y_train_encoded)
print(f"Baseline Gradient Boosting: {round(gb_mod.score(X_test, y_test_encoded), 3)}")
pred_gb = gb_mod.predict(X_test)

# Cross Validate Gradient Boosting Model
scores_gb = cross_val_score(gb_mod, X_test, y_test_encoded, cv=3, scoring="accuracy")
print(f"Scores (Cross Validation) for Gradient Boosting Model: \n{scores_gb}")
print(f"Cross Val Means: {round(scores_gb.mean(), 3)}")
print(f"Cross Val Standard Deviation: {round(scores_gb.std(), 3)}")

'''
# Hyperparameter tuning
params_GB = {
    "n_estimators": [50, 100, 200, 300],  # Number of boosting stages
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
    "max_depth": [3, 5, 7, 10],  # Maximum depth of trees
    "min_samples_split": [2, 5, 10],  # Min samples required to split
    "min_samples_leaf": [1, 2, 4],  # Min samples per leaf
    "subsample": [0.7, 0.8, 0.9, 1.0]  # Fraction of samples used for fitting
}

GridSearchCV_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=params_GB,
    cv=3,
    scoring="accuracy",
    return_train_score=True
)

# Fit model with train data
GridSearchCV_gb.fit(X_train, y_train)

print(f"Best estimator for Gradient Boosting Model:\n{GridSearchCV_gb.best_estimator_}")
print(f"Best parameter values for Gradient Boosting Model:\n{GridSearchCV_gb.best_params_}")
print(f"Best score for Gradient Boosting Model:\n{round(GridSearchCV_gb.best_score_, 3)}")
'''
# Test the new parameter
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=10)
gb_mod = gb.fit(X_train, y_train_encoded)
pred_gb = gb_mod.predict(X_test)

mse_gb = mean_squared_error(y_test_encoded, pred_gb)
rmse_gb = np.sqrt(mse_gb)
scores_gb_train = gb_mod.score(X_train, y_train_encoded)
scores_gb_test = gb_mod.score(X_test, y_test_encoded)

print(f"Mean Square Error for GB = {round(mse_gb, 3)}")
print(f"Root Mean Square Error for GB = {round(rmse_gb, 3)}")
print(f"R^2 (coefficient of determination) on training set= {round(scores_gb_train, 3)}")
print(f"R^2 (coefficient of determination) on testing set= {round(scores_gb_test, 3)}")

print("---------------------------- BEST MODEL SELECTION ----------------------------")
import pickle
# Define models
models_dict = {
    "Logistic Regression": logistic_regression_mod,
    "K-Nearest Neighbors": knn_mod,
    "Random Forest": rf_mod,
    "Gradient Boosting Classifier": gb_mod
}

# DataFrame to store model evaluation results
results_df = pd.DataFrame(columns=["Model", "Train Score", "Test Score", "Precision", "Recall", "F1"])

# Evaluate each model
for model_name, model in models_dict.items():
    train_score = cross_val_score(model, X_train, y_train_encoded, cv=3).mean()
    test_score = model.score(X_test, y_test_encoded)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_encoded, model.predict(X_test),
                                                               average="weighted")

    # Append results to DataFrame
    new_entry = pd.DataFrame([{
        "Model": model_name,
        "Train Score": train_score,
        "Test Score": test_score,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }])

    results_df = pd.concat([results_df, new_entry], ignore_index=True)

# Sort results to get the best model
best_model_info = results_df.sort_values(by=["Test Score", "F1"], ascending=False).iloc[0]

# Retrieve best model
best_model_name = best_model_info["Model"]
best_model = models_dict[best_model_name]

print("\nModel Performance Summary:")
print(results_df.set_index("Model"))
print(
    f"\nBest Model: {best_model_name} (Test Score: {best_model_info['Test Score']:.3f}, F1: {best_model_info['F1']:.3f})")

# Save the best model
model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.sav"
pickle.dump(best_model, open(model_filename, "wb"))

# Load and test saved model
loaded_model = pickle.load(open(model_filename, 'rb'))
prediction_sample = [[1, 2, 78, 56, 76]]  # Example input
print("Predicted Output:", loaded_model.predict(prediction_sample))

print("\n All models and data saved successfully.")