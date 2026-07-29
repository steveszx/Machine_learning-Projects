import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/steveszx/Documents/projects (Github)/amazon_sales_dataset.csv')


#usual checks 
shape = df.shape
#50000 rows, 13 columns of data 

#lets check the data types, split into categorical and continuous variables, descriptive stats

#let convert order_date to datetime 
df["order_date"] =pd.to_datetime(df["order_date"])
#missing value check 
missing = df.isna().sum()
#no missing data 
#split into continous and categorical 
continuous_columns = df.select_dtypes(include=["int64", "float64"])
categorical_columns = df.select_dtypes(include=["str"])

#assess descriptive statistics of the continous variables 
descriptive = continuous_columns.describe().T

#lets check the distribution of total_revenue 
#sns.histplot(df, x="total_revenue")

#the histogram shows that total_revenue is heavily right-skewed with a long-tail on the right and majority of observations on the left
#lets use a boxplot to check for outliers as well 
#sns.boxplot(df, x="total_revenue")

#there are few outliers in total revenue that will be corrected downstream to account for any model that i decide to use 

#let's shift the focus to model_building to predict total_revenue 
"""
baseline model would be linear regression (potentially polynomial regression)
complex model would be a Random forest or xgboost model 
time series model would be theta 
genAI model, use Chronos, all evaluated under the same metrics
"""

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
df = df.sort_values("order_date").reset_index(drop=True)
category_prod = df[df["product_category"]== "Books"]["quantity_sold"]
fig, ax = plt.subplots(figsize=(20,10))
plot_acf(category_prod, lags=20, ax=ax)

#generate a copy of the dataframe for the regression model 

#==================================== liner regression model =========================================
df_reg = df.copy()
#lets drop order_id, order_date, product_id because they offer no predictive value in this model 
df_reg = df_reg.drop(columns=["order_id", "product_id"])

#split df_reg into categorical and continous variables so i can apply one-hot encoder to generate numerical representations
cat_cols = df_reg.select_dtypes(include=["str"]).columns.to_list()
cont_cols =df_reg.select_dtypes(include=["int64", "float64"]).columns.to_list()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, drop='first')

encoded_features = encoder.fit_transform(df_reg[cat_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))
#now i have to concatenate this df with the continous variables 
full_df = pd.concat([df_reg.drop(columns=cat_cols), encoded_df], axis=1)

#now let's do some correlation analysis to determine any highly correlated variables with total_revenue 
corelation = full_df.corr()["quantity_sold"]
#plt.figure(figsize=(20,10))
#corelation.plot(kind="barh", color="coral")

#for our modelling, forecasting sales-revenue using price and whatnot would result in data leakage, 
#the fix is to predict quantity sold and use that to forecast/predict revenue 

#now let's check the time-series plot of quantity_sold


#lemme ascertain the trend of quantity sold using the moving average 
full_df["moving_average_qty"] = full_df["quantity_sold"].rolling(365).mean()
fig, ax = plt.subplots(figsize=(20,10))
sns.lineplot(full_df, x= "order_date", y="quantity_sold", ax=ax)
ax.plot(full_df["order_date"],full_df["moving_average_qty"], linewidth=3)
plt.show()
