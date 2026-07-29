import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#pd.set_option("display.max_columns", None)
"""
This dataset represents synthetic sales data generated for practice purposes only. 
It is not real-time or based on actual business operations, and should be used solely for educational or testing purposes. 
The dataset contains information that simulates sales transactions across different products, regions, 
and customers. Each row represents an individual sale event with various details associated with it.
"""
df = pd.read_csv('/Users/steveszx/Documents/projects (Github)/sales_data.csv')

shape= df.shape
#1000 rows, 14 columns 
# adjust Sale_Date to datetime to eventually enable us to build an effective time series model 
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

# check the number of null values in the dataset 
#print(df.isna().any())
#print(df.isna().sum())

#both checks return that there are no null values in the dataset, which is rare for any dataset however this is synthetic data

#================================== EDA ========================================================
#split the variables between categorical and continous columns and find the necessary descriptive statistics
categorical_columns = df.select_dtypes(include=["str"])
continous_colums = df.select_dtypes(include=["float64", "int64"])
#now that they are split, let's find the descriptive stats on the continous columns 

descriptive_stats = continous_colums.describe().T
#the distribution of sales doesn't necessarily show that there is a great level of skewness, will confirm with an skewness value 
#let's visually confirm outliers using box plots for Sales_Amount and Unit_Price
"""
plt.figure(figsize=(20,10))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
df["Sales_Amount"].plot(kind="box", ax=ax1, title="Boxplot of Sales Amount")
df["Unit_Price"].plot(kind="box", ax=ax2, title="Boxplot of Unit Price")
plt.tight_layout()
"""
#no outliers detected visually and almost looks uniformly distributed, let's confirm uniform distribution

#for the categorical columns let's do some count plots 
Sales_rep = df.Sales_Rep.value_counts()

sns.countplot(df, x= "Region")
#the north and the East appear to have the highest counts within, i will ascertain later whether that actually translates into anything, maybe higher sales 
#or greater market share
sns.countplot(df, x="Payment_Method")
sns.countplot(df, x= "Sales_Channel")
#credit car and bank transfer are the most popular modes of payment

#moving on let's start answering some EDA questions that i will have chatgpt create to guide the analysis 
# 1.Which products and product categories create the most value for the business?
#let's do a groupby product category

category_sales = df.groupby(["Product_Category"]).agg(Mean_Sales= ("Sales_Amount", "mean"))
category_sales.plot(kind="barh", title="Average Sales per Product Category")
#food sales are the largest contributor of sales in this company

region_sales = df.groupby(["Region"]).agg(Mean_sales= ("Sales_Amount", "mean"), 
                                          Mean_cost = ("Unit_Cost", "mean"), 
                                          Count= ("Region", "size")).assign(
                                              Percentage = lambda x: x["Mean_sales"]/ x["Mean_sales"].sum() *100
                                          )
region_sales.plot(kind="bar", title="Average Sales amount per region", figsize=(20,10))
#the south and Northern regions are equal contributors of sales amounts to this company 
#i would like to overlay that analysis by assessing which regions also contribute the most cost 
#the cost component is almost identical accross every region

# 3. Are returning customers more valuable than new customers?
#another grouby to check the average transaction amounts between new and old customers
customers = df.groupby(["Customer_Type", "Sales_Channel"]).agg(Mean_sales= ("Sales_Amount", "mean"))
#returning customers are the highest contributor however their contribuiton is marginal 
#average sales are comprised 49.5% new customers and 50.5% returning customers 
#returning customers purchase more on retail


