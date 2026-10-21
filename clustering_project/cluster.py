import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

"""
Customer Personality Analysis involves a thorough examination of a company's optimal customer profiles. 
This analysis facilitates a deeper understanding of customers, enabling businesses to tailor products to meet the distinct needs, 
behaviors, and concerns of various customer types.

By conducting a Customer Personality Analysis, businesses can refine their products based on the preferences of specific customer segments.
Rather than allocating resources to market a new product to the entire customer database, 
companies can identify the segments most likely to be interested in the product. 
Subsequently, targeted marketing efforts can be directed toward those particular segments, 
optimizing resource utilization and increasing the likelihood of successful product adoption.

"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Explore and cluster customer profiles.")
parser.add_argument(
    "--data", type=Path,
    default=Path(__file__).with_name("customer_segmentation.csv"),
    help="Path to the customer segmentation CSV (not included in this repository).",
)
args = parser.parse_args()
df = pd.read_csv(args.data)

#inspect the data set adn the data types and ascertain whether anything is out of place #
# df.info()

#check for missing data 
missing = df.isna().sum()
#print("the following shows the missing observations for each variable")


#income has got 24 missing values, let's rectify that 
#visualise the distribution of income first using a boxplot and histogram 
fig, ax = plt.subplots(figsize=(20,10))
sns.histplot(data=df, x="Income", ax=ax, bins =10)
plt.title("Histogram showing distribution of Income")
plt.tight_layout()


#income is heavily right skewed, with a majority of the incomeobservations clustered in the lower regions 
# Use the median to reduce the influence of extreme incomes on imputation.

from sklearn.impute import SimpleImputer
simp_imp = SimpleImputer(strategy="median")
df["Income"] = simp_imp.fit_transform(df[["Income"]])

#income has been fixed 
#========================================SUMMARY STATISTICS ============================================
summary_stats = df.describe().T


#=========================================EDA===========================================================
#For the purposes of any analysis, i will drop the ID column 
df = df.drop(columns=["ID"])


#how does income relate to total spending and product preferences 
#to find total spending i need to sum up the total spending on individual items 
columns = df.columns.to_list()

df["total_spend"]= df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)


#compare to total spend to income levels 
#let's do a scatterplot between total spend and income 

fig, ax1 = plt.subplots(figsize=(20,10))
sns.scatterplot(data=df, x="Income", y="total_spend", ax=ax1)
plt.tight_layout()
plt.title("Scatter Plot for Income and Total Spend")
#the graph shows there's a postive association between total spending and income levels however there isn't a clear 
#cut relation showing that higher income levels correspond to more spending 

#to add labels onto the various income groups, design a custom function 

df["Income_group"] = pd.qcut(df["Income"], q=3, labels = ["Low Income", "Middle Income", "Higher Income"])


#df["Year_Birth"] = pd.to_datetime(df["Year_Birth"], format="%Y")
#age might be a much more useful metric to have in this dataset 
reference_year= 2025
import datetime as dt
df["Age"] = reference_year - df["Year_Birth"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["Customer_tenure"] = reference_year - df["Dt_Customer"].dt.year
#print(df.head(10))
#the aim of this clustering exercise is to cluster customers based on their purchasing patterns and demographic of said customers 
#+=================================================FEATURE SELECTION ================================================
#assess whether Z_costContact and Z_revenue have any meaningful addition to the dataset 
#print(df.Z_CostContact.unique())
#print(df.Z_Revenue.unique())

#both these variables have got constant values hence they don't really offer any predictive/clustering value since it's the same for eachh 
#customer 

#split variables into categorical and numerical to allow for encoding where necessary 
categorical_variables = df.select_dtypes(include=["object", "category"]).columns.to_list()

from sklearn.preprocessing import OneHotEncoder, RobustScaler
encoder = OneHotEncoder(sparse_output=False)

encoded_features = encoder.fit_transform(df[categorical_variables])
df.info()
#select all numerical variables except the binary variables
numerical_variables = df.select_dtypes(include=["int64", "float64", "int32"])
numerical_variables = numerical_variables.drop(columns=["Year_Birth","AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3","AcceptedCmp4","AcceptedCmp5", "Response", 
                                                        "Complain", "Z_CostContact", "Z_Revenue"])
num_cols = numerical_variables.columns.to_list()
scaler = RobustScaler()
scaled_cols = scaler.fit_transform(df[num_cols])
binary_cols = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5", "Response", "Complain"
]
scaled_df = pd.DataFrame(scaled_cols, columns= scaler.get_feature_names_out(num_cols), index= df.index)
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_variables), index=df.index)
full_df = pd.concat([scaled_df, encoded_df, df[binary_cols]], axis=1)
full_df = full_df.drop(columns=["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
                                "MntGoldProds", "Income_group_Higher Income", "Income_group_Low Income", "Income_group_Middle Income"])
#these variables are being dropped because they are already represented by the total spend and income values thus including 
#them twice increases their influence which may negatively overstate their importance 

#==================================BASELINE K-MEANS CLUSTERING==============================================
from sklearn.cluster import KMeans
#let's find the ideal K using the elbow method 
wcss= []
for i in range(1,21):
    kmeans_try = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans_try.fit(full_df)
    wcss.append(kmeans_try.inertia_)

plt.figure(figsize=(20,10))
plt.plot(range(1,21), wcss, marker="o", linestyle="--")
plt.title("NUMBER OF CLUSTERS USING THE ELBOW METHOD")
plt.xlabel("Clusters")
plt.ylabel("WCSS")

#the pivot in the graph happens at 2.5 thus i will select 3 clusters in my first iteration 
K_men1 = KMeans(n_clusters=3, init='k-means++', random_state=42)
K_men1.fit(full_df)

full_df["Kmen1_labels"] = K_men1.labels_
df["Kmen1_labels"] = K_men1.labels_
# for the sake of visualisation here, i will plot a scatter diagram of Education and Income 
fig, ax2 = plt.subplots(figsize=(20,10))
sns.scatterplot(data=df, x="Age", y="Income", hue="Kmen1_labels", ax=ax2)
plt.tight_layout()
plt.title("Scatter plot of Age and Income highlighting Cluster Seperation")
#group by clusters 
sub_df = df.groupby(["Kmen1_labels"])[["Income", "total_spend", "Age"]].median()



#================================================PCA + KMEANS=============================================
#exclude KMen_lables from full_df
full_df2 = full_df.drop(columns=["Kmen1_labels"])

from sklearn.decomposition import PCA 
pca = PCA()
#since i dont know many principal components would be ideal, i start by initialising it empty then select based on the 
#explained_variance ratio graph 
pca.fit(full_df2)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#graph of cummulative expalined varaince
fig, ax3 = plt.subplots(figsize = (20,10))
ax3.plot(range(1,len(cumulative_variance)+1),cumulative_variance , marker ="o", linestyle="--")
plt.tight_layout()
ax3.set_title("Explained Variance by Components")
ax3.set_xlabel("Number of Components")
ax3.set_ylabel("Cumumlative Variance Explained")

#from the graph the number of principal components that retain about 70-80% is around 10 components 
pca2 = PCA(n_components=10)
scores_pca = pca2.fit_transform(full_df2)

#now determine the optimal K using these PCA components 
wcss_2 = []
for i in range(1,21): 
    Kmen2_try = KMeans(n_clusters=i, init="k-means++", random_state=42)
    Kmen2_try.fit(scores_pca)
    wcss_2.append(Kmen2_try.inertia_)

fig, ax4 = plt.subplots(figsize =(20,10))
ax4.plot(range(1,21), wcss_2, marker="o", linestyle="--")
ax4.set_title("ELBOW METHOD USING PCA SCORES")
ax4.set_xlabel("Clusters")
ax4.set_ylabel("WCSS")


#2 possible candidate options clusters 3 or 4 clusters 
KMen2 = KMeans(n_clusters=4, init="k-means++", random_state=42)
KMen2.fit(scores_pca)
df["KMen2_labels"] = KMen2.labels_
new_df = pd.concat([df, pd.DataFrame(scores_pca)], axis=1)
new_df.info()
#let's visualise the first 2 principal components on a scatter plot 

fig, ax5 = plt.subplots(figsize=(20,10))
sns.scatterplot(data=new_df, x=0, y =1, hue="KMen2_labels", ax=ax5)
plt.tight_layout()
plt.title("Scatter Plot of first and Second Principal Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

KMen3 = KMeans(n_clusters=3, init="k-means++", random_state=42)
KMen3.fit(scores_pca)
new_df["KMen3_labels"] = KMen3.labels_
fig, ax6 = plt.subplots(figsize=(20,10))
sns.scatterplot(data=new_df, x=0, y =1, hue="KMen3_labels", ax=ax6)
plt.tight_layout()
plt.title("Scatter Plot of first and Second Principal Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")


#visually 3 clusters shows more seperation with little overlap between clusters, however let's confirm by 
#calculating the silhoutte score for 4 clusters and 3 clusters seperately 
from sklearn.metrics import silhouette_score
sil_3 = silhouette_score(scores_pca, KMen3.labels_)
sil_4 = silhouette_score(scores_pca, KMen2.labels_)
base_sil = silhouette_score(full_df2, K_men1.labels_)
print(f"3 Cluster silhoutte score = {np.round(sil_3,3)}, 4 Cluster silhoutte score = {np.round(sil_4,3)},\
      relative to baseline silhoutte score = {np.round(base_sil, 3)}")
# Compare all label sets in the same feature space as well.
print("PCA labels evaluated in original preprocessed space:", {
    "3 clusters": round(silhouette_score(full_df2, KMen3.labels_), 3),
    "4 clusters": round(silhouette_score(full_df2, KMen2.labels_), 3),
})

#3 clusters have got a higher silhoutte with more cluster seperation hence that one is chosen 
spending_profile = new_df.groupby(["KMen3_labels"])[["Income", "total_spend", "Age"]].median()
print(spending_profile.reset_index())

#cluster 1 is low income, low spend cluster with relatively younger customers 
#cluster 2 is high income, high spend cluster with middle aged customers 
#cluster 3 sits in the middle 

# Customer profiles use original units and unscaled categorical indicators.
# Keep product spending here to describe preferences, even though it was
# excluded from the clustering inputs.
profile_numeric_cols = num_cols.copy()
profile_category_cols = [
    column for column in encoded_df.columns
    if not column.startswith("Income_group_")
]
profile_binary_cols = profile_category_cols + binary_cols

profile_df = pd.concat(
    [
        df[profile_numeric_cols],
        encoded_df[profile_category_cols],
        df[binary_cols],
        new_df[["KMen3_labels"]],
    ],
    axis=1,
).copy()

profile_features_num =[
    "Income", 
    "Recency", 
    "NumDealsPurchases", 
    "NumWebPurchases",
    "NumCatalogPurchases",
    "total_spend", 
    "Age", 
    "Customer_tenure"
]
profile_binary = [
    "Education_2n Cycle",
    "Education_Basic", 
    "Education_Graduation", 
    "Education_Master", 
    "Education_PhD",
     "Kidhome",
    "Teenhome"
]

cluster_summary = profile_df.groupby(["KMen3_labels"])[profile_features_num].median().round(2)
cluster_summary_2 = profile_df.groupby(["KMen3_labels"])[profile_binary].mean().round(2)
full_summary = pd.concat([cluster_summary, cluster_summary_2], axis=1)

column_range = full_summary.max() - full_summary.min()
heatmap_normalised = (full_summary - full_summary.min()) / column_range.replace(0, 1)
# Normalize the data column by column (0 to 1 scale) so colors map correctly across mixed units

fig, ax7 = plt.subplots(figsize=(20,10))
sns.heatmap(heatmap_normalised, cmap = sns.cubehelix_palette(as_cmap=True), annot = full_summary, fmt= ".2f")
plt.title("Heatmap of Clusters")
plt.ylabel("Strategic Archetype")
plt.xlabel("Behavioural features")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
