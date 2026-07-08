import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df= pd.read_csv('creditcard.csv')
#the aim of this model in this dataset is to build a classification model for identifying which transactions are fradulent 
#print(df.head(10))

#EDA ANALYSIS WHERE POSSIBLE 
#The v1 -v28 variables are PCA components thus they have scaled already
#Descriptive statstics focus is on Amount as it's the only variable that provides some relatively good information 

#print(df[["Amount", "Time"]].describe())

#distribution of Amount in the dataset visually 
"""
plt.figure(figsize=(20,10))
sns.histplot(data=df, x= "Amount", hue="Class", color= "green")
plt.tight_layout()
plt.xlabel("Distribution of Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Amount Variable")
#the graph suggests that is heavily right_skewed (mean > median) with a long right tail with a few observations being on the outskirts 
plt.figure(figsize=(20,10))
sns.histplot(data=df, x= "Time", hue="Class", color= "green")
plt.tight_layout()
plt.xlabel("Distribution of Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Amount Variable")
"""


#time shows an interesting distribution, with no clear basis for it being skewed however, it seems that it has 2 seperate distribution present 
#almost looks like 2 normal distributions. Will look further into this


#lets confirm the skewness of this via .skew()
"""
skewness_amount = df.Amount.skew()
skewness_time = df.Time.skew()
print(f"Skewness of Transaction Amount = {round(skewness_amount,2)}, Skewness of Time variable = {round(skewness_time, 2)}")
"""

#CORRELATION ANALYSIS TO CHECK WHETHER ANY VARIABLES CAN BE REMOVED BECAUSE IT'S TOO CLOSELY RELATED TO THE TARGET 
plt.figure(figsize=(20,10))
correlation = df.corr(numeric_only=True)["Class"].sort_values(ascending=True)
correlation.plot(kind="barh", title="Correlation of Variables Vs Target")

#from this graph there's no indication of any variable that is heavily correlated with the target thus they all remain relevant in our classification problem


from sklearn.model_selection import train_test_split
#let's split our dataset into train and test sets 
X = df.drop(columns=["Class"])
Y = df.Class
#30/70 split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.preprocessing import RobustScaler
rs_time = RobustScaler()
rs_amount = RobustScaler()
X_train["Time"] = rs_time.fit_transform(X_train[["Time"]])
X_train["Amount"] = rs_amount.fit_transform(X_train[["Amount"]])

#reuse the scalers fit on training data - refitting on test data would leak test-set distribution info
X_test["Time"] = rs_time.transform(X_test[["Time"]])
X_test["Amount"] = rs_amount.transform(X_test[["Amount"]])


#define the pipeline
#pipeline will take steps, the SMOTE, logistic regression model as well 


from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
steps = [("SMOTE", SMOTE()), ("logis", LogisticRegression(random_state=42))]

"""
Logistic_pipeline = Pipeline(steps=steps, verbose=True)

#define the parameter grid with the model_underscoring the model and it's parameters 
#let's rewrite the previous paramter grid and then tailor to work in concert here 
common_params = {"logis__tol": [0.001, 0.1, 0.5],
              "logis__C": np.logspace(-4,4,7), 
              "logis__max_iter": [100, 250, 500]}
#this is the normal parameters now i need to tweak it 
grid_params = [{**common_params, "logis__solver": ["lbgfs"], "logis__penalty": ["l1"]},
               {**common_params, "logis__solver": ["liblinear"], "logis__penalty": ["l1", "l2"]},
               {**common_params, "logis__solver": ["saga"], "logis__penalty": ["l1", "l2"]},
               {**common_params, "logis__solver": ["saga"], "logis__penalty":["elasticnet"], "logis__l1_ratio": [0.2, 0.5, 0.8]}]

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = Logistic_pipeline, param_grid=grid_params, cv=5, verbose=True, n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(grid_search.best_estimator_)
"""
#so some of the solver models work with specific penalties 
#perform an initial grid search with the parameters 
#get the best model with the best parameters 

"""
Pipeline(steps=[('SMOTE', SMOTE()),
                ('logis',
                 LogisticRegression(C=np.float64(0.002154434690031882),
                                    max_iter=500, penalty='l1', random_state=42,
                                    solver='liblinear', tol=0.5))],
         verbose=True)
"""

#now with the best possible parameters let's rebuild pipeline 
steps_2 = [("SMOTE", SMOTE(random_state=42)), ("logis", LogisticRegression(C=0.002154434690031882, max_iter=500, penalty="l1", 
                                                                           random_state=42, solver="liblinear", tol=0.5, l1_ratio=0.0))]

pipe_2 = Pipeline(steps=steps_2)
pipe_2.fit(X_train, Y_train)
predictions = pipe_2.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class_report = classification_report(Y_test, predictions)
conf_matrix = confusion_matrix(Y_test, predictions)

print("Logistic Regression")
print(class_report)

#visualise the confusion matrix
vis = ConfusionMatrixDisplay(conf_matrix)
vis.plot()


#few things to note 
#the models predictive capabilities in identifying the positve class (non-fraud cases) is really high evidenced by the accuracy of 99%
#the precision of the model which measures "out of all the transactions flagged by the model, how many were actually positve", in this case out of all the
#fraud transactions how many were actually fraudulent cases. if precision is low then legit transactions are being flagged as fraudulent 

#recall on the other hand measures the model predictive ability, of all the actual frauds how many times did the model catch, and in this case recall is high for both fraudulent and non-fraudulent cases 
#implying that the model is effective at identifying both fraudulent and non-fraudulent cases
#low precision, high recall is desirable because the model catches catches legit signs of fraud however there are many false alarms as a result of low precision.


#to truly assess performance, let's fit another classifier 

#============================RANDOM FOREST CLASSIFIER ==========================================
from sklearn.ensemble import RandomForestClassifier
#note that decision trees are robust to outliers so if that exists within the data, it will be swiftly handled 

#hyper parameter tuning of random_forest model 
"""
steps_3 = [("SMOTE", SMOTE(random_state=42)), ("RF", RandomForestClassifier(random_state=42))]
#similarlly let's perform a Grid search CV for the optimal parameters of this random_forest classifier 

#define a pipeline for the random_forest 
Random_forest_pipeline = Pipeline(steps=steps_3)
grid_params_3 = {"RF__n_estimators": [50, 100, 150],
                 "RF__max_depth": [3, 5, 7],
                 "RF__min_samples_split": [2, 3, 4], 
                 "RF__min_samples_leaf": [1, 3, 5]}

#this is the grid to search 
from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator=Random_forest_pipeline, param_grid=grid_params_3, n_jobs=-1, cv=5, verbose=True)
grid_model.fit(X_train, Y_train)
print(grid_model.best_estimator_)

Pipeline(steps=[('SMOTE', SMOTE(random_state=42)),
                ('RF',
                 RandomForestClassifier(max_depth=7, min_samples_split=4,
                                        n_estimators=150, random_state=42))])
"""
steps_4 = [("SMOTE", SMOTE(random_state=42)), ("RF", RandomForestClassifier(max_depth=7, min_samples_split=4,
                                                                             n_estimators=150, random_state=42))]
pipe_4 = Pipeline(steps=steps_4)
pipe_4.fit(X_train, Y_train)
prediction_rf = pipe_4.predict(X_test)

class_report_2 = classification_report(Y_test, prediction_rf)
print("Random Forest")
print(class_report_2)

#the random forest has a higher f1 score coupled with a higher precision while maintaing a relatively high recall value meaning the model
#is effectively of capturing fraudulent and non-fraudelent cases and the higher precision means that the model cries wolf less. 

