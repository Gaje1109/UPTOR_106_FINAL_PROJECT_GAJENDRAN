import pickle
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error

# Import the CSV file
fetal_df= pd.read_csv("/MediPredict_AI_Health_Analyzer/CSVFiles/fetal_health.csv")

print("Dataframe loaded and opened: \n", fetal_df.head())

#print("Information about the data set: \n", fetal_df.info())

#sns.pairplot(fetal_df, hue = "fetal_health")
#plt.savefig("/MediPredict_AI_Health_Analyzer/Images/FHP_patterns.png")
#plt.show()

#print("Columns List: ", fetal_df.columns)



# converting all the attributes into a standard way betweeen range -1 to 1

fetal_columns= ['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency']

scale_X= StandardScaler()
X= pd.DataFrame(scale_X.fit_transform(fetal_df.drop(["fetal_health"], axis =1)), columns= fetal_columns)


Y = fetal_df["fetal_health"]
print("X Head: ", X.head())
print(" Y is :", Y)

# Importing train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("X_train: \n",X_train.shape)
print("X_test: \n",X_test.shape)

print("Y_train: \n",y_train.shape)
print("Y_test: \n",y_test.shape)

print("---------------------------------------------GRADIENT BOOSTING-----------------------------------------------------------------------")
'''--------------------------------------------------------------------------------------------------------------------'''
# Initialize and train the Gradient Boosting Classifier model
gb_classifier= GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

# Make Predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Calculate Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
f1= f1_score(y_test, y_pred, average='weighted')

# Calculate AUC for multiclass
y_scores = gb_classifier.predict_proba(X_test)
#auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {fscore}")
#print(f"AUC: {auc:.2f}")

pickle.dump(gb_classifier, open('../Model/fetal_health_classifier.sav', 'wb'))

print("\n All models and data saved successfully.")