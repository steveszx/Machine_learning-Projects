import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels

#train logistic regression as a baseline and compare to Random forest classifier

df = pd.read_csv('/Users/steveszx/Documents/projects (Github)/creditcard.csv')
#the aim of this model in this dataset is to build a classification model for identifying which transactions are fradulent 
#print(df.head(10))

#EDA ANALYSIS WHERE POSSIBLE 
#The v1 -v28 variables are PCA components thus they have scaled already
#Descriptive statstics focus is on Amount as it's the only variable that provides some relatively good information 

#print(df[["Amount", "Time"]].describe())

#distribution of Amount in the dataset visually 
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

#time shows an interesting distribution, with no clear basis for it being skewed however, it seems that it has 2 seperate distribution present 
#almost looks like 2 normal distributions. Will look further into this


#lets confirm the skewness of this via .skew()
"""
skewness_amount = df.Amount.skew()
skewness_time = df.Time.skew()
print(f"Skewness of Transaction Amount = {round(skewness_amount,2)}, Skewness of Time variable = {round(skewness_time, 2)}")
"""

#check for class imbalance 
#print(df.Class.value_counts(normalize=True))
#so the dataset has got a very high class imbalance in the dataset, with 99.98% being accurate transactions (0) 
# and 0.01% being fraudulent transactions, the implication is that our model will always most likely predict the 0 class due to this class imbalance

#remember split into train and test before scaling variables to prevent any data leakage

#let's split our dataset into train and test sets 
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Class"])
Y = df.Class
#30/70 split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#now let's scale Amount and Time using Scaler 
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
rs2 = RobustScaler()
X_train["Time"] = rs.fit_transform(X_train[["Time"]])
X_train["Amount"] = rs.fit_transform(X_train[["Amount"]])

#fit a different scaler to the test data 
X_test["Time"] = rs2.fit_transform(X_test[["Time"]])
X_test["Amount"] = rs2.fit_transform(X_test[["Amount"]])

#all variables have been scaled 
#ADDRESSING CLASS IMBLANCE USING SMOTE (OVERSAMPLING MINORITY CLASS) TBC
# https://medium.com/data-science/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a
#USING SMOTE FIRST THEN COMPARE WITH RANDOM-OVERSAMPLING TO ESTIMATE WHICH PROVIDES BETTER RESULTS 
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
#then fit the smote on the X_train and Y_train 
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
#when i check value count's smote has effectively removed any class imbalance in the dataset, let's go ahead and build our first model 
#print(Y_train_smote.value_counts())

#let's build a logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#let's implement a grid_search to find out the best parameters to use to tune this logistic regression model 
"""
from sklearn.model_selection import GridSearchCV
#so to implement the grid search, let's make it 2 fold
common_params = {"tol": [0.001, 0.05, 0.1], 
               "C": np.logspace(-4,4,10),
               "max_iter"  : [100,1000]}

grid_values = [{**common_params, "solver": ["lbfgs"], "penalty": ["l2"]}, 
               {**common_params, "solver": ["liblinear"], "penalty": ["l1", "l2"]},
               {**common_params, "solver": ["saga"], "penalty": ["l1", "l2"]},
                {**common_params, "solver": ["saga"], "penalty": ["elasticnet"], "l1_ratio": [0.2, 0.5, 0.8]}]

clf = GridSearchCV(estimator=lr, cv=5, param_grid=grid_values, verbose=True, n_jobs=-1)
best_model = clf.fit(X_train_smote, Y_train_smote)
print(best_model.best_estimator_)

#np.logspace generates evenly spaced numbers on a logarithmic scale, 1st argument is 10^, last argument= number of values generated
"""
#the best parameters found were (C= 166.81), penalty = "l2", solver ="liblinear", tol=0.0001
lr = LogisticRegression(C=166.81, solver="liblinear", penalty="l2", tol=0.001)
lr.fit(X_train_smote,Y_train_smote)

#now let's assess the model 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score,classification_report
Y_probabilities = lr.predict_proba(X_test)
y_pred = lr.predict(X_test)
"""
for i, prob in enumerate(Y_probabilities): 
    print(f"Observation {i+1} probability of Class 0 = {round(prob[0])}, probability of Class 1 = {round(prob[1])}")
"""
accuracy = accuracy_score(y_true= Y_test, y_pred=y_pred)
#this model has got a quite high accuracy, slightly alarming 
#let's create a confusion matrix 

confusion_matrix1 = confusion_matrix(y_true=Y_test, y_pred=y_pred)
vis = ConfusionMatrixDisplay(confusion_matrix1)
plt.figure(figsize=(20,10))
vis.plot()
plt.tight_layout()


f1_val = f1_score(Y_test,  y_pred)
classifier = classification_report(Y_test, y_pred)
print(classifier)

#precision = purity of alerts. out of every transaction flagged as fraud, how many of them were actually fradulent cases. Precison focuses on predictions. if precision is high that means the model identifies a lot of fraud cases, it if it;s low legit transactions are flagged as fraudulent 

#recall= of all the actual frauds, how much did my model catch. retrieval rate. low recall = fraud passes undeteceted, in this case recall is high for both classes thus the model is doing well in identifying fraudelent and non-fraudulent cases. recall is about reality. 

#low precision, high recall is desirable because the model catches catches legit signs of fraud however there are many false alarms as a result of low precision. 

#i want to rebuild this script using pipeline to mitigate any data leakage concerns 

