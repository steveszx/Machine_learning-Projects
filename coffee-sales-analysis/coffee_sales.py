import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('/Users/steveszx/Documents/projects (Github)/index_1.csv')

#convert the date to datetime 
df["date"] = pd.to_datetime(df['date'])
df["datetime"] = pd.to_datetime(df["datetime"])
#check the data types of columns 
#df.info()

#Drop one of the datetime columns 
df = df.drop(columns=["datetime", "card"])
#the card column offers no predictive value at this stage of modelling thus it would be dropped 
#critical check: missing data columns 
"""
print(df.isna().sum())
"""

#Exploratory Data Analysis 
#Average coffee sales per coffee_name or coffee brand, check how many unique coffee brands exist in the dataset 
average_sales = df.groupby(["coffee_name"]).agg(Mean_Sales = ("money", "mean")).reset_index()

#the highest average sales category would be Hot Chocolate 
#let's check how much data we have for each category of coffee 
df.coffee_name.value_counts(normalize=True).round(2)
daily_volume = df.groupby(["date"]).size()
daily_volume.describe()
#this shows that on average there isn't enough transaction volume per day such that each category is sold effectively everyday thus i will create a more 
#global model including all the time series variables 

df.money.plot()

#the plot of the target doesn't reveal much about the nature of the series, whether seasonality may or may not exist

df= df.rename(columns= {"date": "ds", "money": "y"})
df = df.drop(columns=["cash_type"])



#approach is to aggregate the daily sales per date so each date has one unique sales value
#my initial approach was working with hourly date because the datetime index highlighted that transactions were made at multiple times in the day 

daily_sales_revenue = df.groupby(["ds"]).agg(Daily_sales =("y", "sum")).reset_index()

#now with this inplace, let's create a temporary dataframe where date is the index 
df_temp = daily_sales_revenue.copy()
df_temp.set_index("ds", inplace=True)

df_temp.plot(ylabel="Sales", figsize=(20,10))

#this graph highlights that some seasonality may exist within this coffee sales dataset

#let's check monthly seasonality of coffee sales using month_plot 

#alright let's move to decompose this daily sales revenue dataframe 
decomposition = seasonal_decompose(x=df_temp, model="multiplicative", period=7)
fig = decomposition.plot()
fig.set_size_inches((20,10))



#this decomposition doesn't reveal much about the seasonality present in the data, the residuals however reveal some level of
#heteroscedasticity may be present in the data, evidenced by the fanning out as time goes by

#no need to cehck for stationarity as i will be applying the prophet model
plot_pacf(df_temp)

#some lags might prove useful for this dataset

lags = [1,2,3]
for var in lags: 
    daily_sales_revenue[f"lag_{var}"] = daily_sales_revenue.Daily_sales.shift(var)

daily_sales_revenue = daily_sales_revenue.dropna()

#now time to rebuild a prophet model to forecast daily sales 
daily_sales_revenue = daily_sales_revenue.rename(columns={"Daily_sales":"y"})

#prophet is widely known for its ability to handle holiday and seasonality effects thus
#thus i am going to print all the unique dates in my dataset and them as relevant holidays 
dates = pd.to_datetime(daily_sales_revenue.ds.unique())

#now lets import the holidays package 
import holidays
#the dataset gives no indication of location so i will be using the US as a proxy for location
country_holidays = holidays.US(years=range(dates.min().year, dates.max().year + 1))

#now create a new dataframe with all the dates and includes the relevant holiday columns 
holiday_df = pd.DataFrame({"date": dates})
holiday_df["holiday"] = holiday_df["date"].map(lambda d: country_holidays.get(d))
holiday_df["is_holiday"] = holiday_df["holiday"].notna()

holi_days =(holiday_df[holiday_df["is_holiday"]])
holi_days = holi_days.rename(columns={"date": "ds"}).drop(columns="is_holiday")

periods = 30
train = daily_sales_revenue[:-periods]
test = daily_sales_revenue[-periods:]

#now let's do some parameter tuning to find the best for the prophet model 
from sklearn.model_selection import ParameterGrid
param_grid = {"seasonality_mode": ["additive", "multiplicative"], 
          "seasonality_prior_scale": [10.0, 15.0], 
          "changepoint_prior_scale": [0.01,0.05], 
          "holidays_prior_scale": [5, 10, 15]
}
all_params = list(ParameterGrid(param_grid=param_grid))
from prophet import Prophet

"""
if __name__ == "__main__": 
    rmse_list = []
    mape_list = []

    for params in all_params: 
        m = Prophet(yearly_seasonality=False, 
                    weekly_seasonality=True, 
                    holidays= holi_days, 
                    **params)

        m.fit(train)

        #now moving cross validation
        from prophet.diagnostics import cross_validation, performance_metrics
        df_cv = cross_validation(m, initial="200 days", period = "100 days", horizon="100 days", parallel="processes")
        rmse = round(performance_metrics(df_cv)["rmse"].mean(),2)
        mape = round(performance_metrics(df_cv)["mape"].mean()*100,2)

        rmse_list.append(rmse)
        mape_list.append(mape)

    outcome = pd.DataFrame(all_params)
    outcome["rmse"] = rmse_list
    outcome["mape"] = mape_list

    best_params = all_params[rmse_list.index(min(rmse_list))]
    print(best_params)

"""

#{'changepoint_prior_scale': 0.01, 'holidays_prior_scale': 10, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 10.0}

#after running the parameter grid, the best parameters have been discovered thus, i have to run the prophet model again with these new parameters 
m = Prophet(yearly_seasonality=False, 
            weekly_seasonality=True,
            changepoint_prior_scale= 0.1, 
            holidays_prior_scale= 10, 
            seasonality_mode= "additive", 
            seasonality_prior_scale= 10.0, 
            holidays= holi_days)
m.fit(train)
predict = m.predict(test)

#mathematical calculation of MAPE and RMSE 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
mse = mean_squared_error(test["y"], predict["yhat"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(test["y"], predict["yhat"])
mape = mean_absolute_percentage_error(test["y"], predict["yhat"])* 100
#print (f"MAPE = {np.round(mape,2)}, MAE = {np.round(mae,2)}, RMSE = {np.round(rmse, 2)}")


#now i will move on to visualising the predictions to see how the prophet model did with this dataset 
plt.figure(figsize=(20,10))
plt.plot(test["ds"], test["y"], color = "red", label= "test_data")
plt.plot(predict["ds"], predict["yhat"], color = "green", label= "predictions")
plt.fill_between(predict["ds"], predict["yhat_upper"], predict["yhat_lower"], color="pink")
plt.legend()
plt.tight_layout()


#the prophet model does poorly at predicting the test data, could be as a result of structural breaks in the data that is not present in the train data 
#the model could also potentially be overffiting on the train data as a result of limited training data of around 350 observations.


#build an alternative simpler baseline model such as an SARIMA model 
from statsmodels.tsa.statespace.sarimax import SARIMAX
#SARIMAX model usuallly requires the target to be stationary, which can be checked using the adfuller test 
from statsmodels.tsa.stattools import adfuller
adf_test_1 = adfuller(daily_sales_revenue["y"])

#p-value is less than 1, thus the null hypothesis is rejected (null is that the series is not stationary)

#no need for differencing in this case then 

#so now i will set up my SARIMAX model, but i will first perform hyperparameter tuning to find the optimal parameters of the SARIMAX model 
#let's use the mango python package for hyperparameter tuning
from mango import scheduler, Tuner

from sklearn.metrics import mean_squared_error
def sub_sarima_objective_function(args_list):
    global data
    n = len(data)
    params_eval = []
    results = []
    max_horizon = 30 #forecast 14 days ahead
    n_folds = 3
    fold_step = 15 #spacing between successive origins
    last_origin = n - max_horizon
    first_origin = last_origin - (n_folds - 1) * fold_step

    for params in args_list:
        try:
            p, d, q = params["p"], params["d"], params["q"]
            P, D, Q = params["P"], params["D"], params["Q"]
            trend = params["trend"]

            fold_mses = []
            for origin in range(first_origin, last_origin + 1, fold_step):
                train_fold = data[:origin]
                model2 = SARIMAX(np.log(train_fold), order=(p,d,q), seasonal_order=(P, D, Q, 7), trend=trend).fit()
                predictions = model2.forecast(steps=max_horizon)
                pred_exp = np.exp(predictions)
                y_true = data.iloc[origin: origin + max_horizon]
                fold_mses.append(mean_squared_error(y_true=y_true, y_pred=pred_exp))

            params_eval.append(params)
            results.append(np.mean(fold_mses))
        except:
            params_eval.append(params)
            results.append(1e5)
    return params_eval, results

param_space =dict(p = range(0,10), 
                  d = range(0,3), 
                  q = range(0,10), 
                  P = range(0,4), 
                  D = range(0,2),
                  Q = range(0,4), 
                  trend = ["n", "c", "t", "ct"])
conf_dict = dict()
conf_dict["num_iteration"] =75

data = train["y"]

#tuner = Tuner(param_space, sub_sarima_objective_function, conf_dict)

#results = tuner.minimize()
#print(f"best Parameters ={results['best_params']}")
#print(f"best loss = {results['best_objective']}")
#best Parameters ={'D': 1, 'P': 2, 'Q': 1, 'd': 1, 'p': 1, 'q': 5, 'trend': 't'}""" log model 

#============================== normal model =================================================
model2 = SARIMAX(endog=train["y"], order = (4,2,3), seasonal_order=(4,0,6,7), trend="c").fit()
predicts = model2.forecast(steps = periods)
plt.figure(figsize=(20,8))
plt.plot(train["ds"], train["y"], label = "train_data", color ="red")
plt.plot(test["ds"], test["y"], label="test_data", color= "blue")
plt.plot(test["ds"], predicts, label="forecast", color ="green")


#let's calculate some error metrics, MAPE, RMSE AND MAE 
mape2 = np.round(mean_absolute_percentage_error(y_true= test["y"], y_pred= predicts),2)
mae2 = np.round(mean_absolute_error(y_true=test["y"], y_pred= predicts),2)
print(f"MAPE = {mape2*100}%")
print(f"MAE = {mae2}")
#=================================== log model ==============================================
#try modeling on log(y) instead to stabilize the growing variance flagged in the decomposition residuals
train_log = np.log(train["y"])
model3 = SARIMAX(endog=train_log, order=(1,1,5), seasonal_order=(2,1,1,7), trend="t").fit()
predicts_log = model3.forecast(steps=periods)
predicts_exp = np.exp(predicts_log)

plt.figure(figsize=(20,8))
plt.plot(train["ds"], train["y"], label="train_data", color="red")
plt.plot(test["ds"], test["y"], label="test_data", color="blue")
plt.plot(test["ds"], predicts_exp, label="log-transformed forecast", color="green")
plt.legend()
plt.tight_layout()

mape3 = np.round(mean_absolute_percentage_error(y_true=test["y"], y_pred=predicts_exp), 2)
mae3 = np.round(mean_absolute_error(y_true=test["y"], y_pred=predicts_exp), 2)
print(f"Log-transformed SARIMA MAPE = {mape3*100}%")
print(f"Log-transformed SARIMA MAE = {mae3}")

#conclusion: log-transform performed worse than the raw-scale model across every order tried (including a properly
#multi-fold-tuned one), not better as hypothesized. Likely cause: a linear trend fit in log-space 

#kept here as documented exploratory work; the raw-scale SARIMA above remains the working baseline.

#check whether the test period is different from train (possible structural break in the data)
print("train stats:")
print(train["y"].describe())
print("test stats:")
print(test["y"].describe())

#visualize rolling mean/std across the full series to see if the shift is gradual or a sharp jump at the split
rolling_window = 7
rolling_mean = daily_sales_revenue["y"].rolling(rolling_window).mean()
rolling_std = daily_sales_revenue["y"].rolling(rolling_window).std()
split_date = test["ds"].iloc[0]

fig, ax = plt.subplots(2, 1, figsize=(20,10), sharex=True)
ax[0].plot(daily_sales_revenue["ds"], rolling_mean, label=f"{rolling_window}-day rolling mean")
ax[0].axvline(split_date, color="black", linestyle="--", label="train/test split")
ax[0].set_ylabel("Rolling Mean")
ax[0].legend()

ax[1].plot(daily_sales_revenue["ds"], rolling_std, label=f"{rolling_window}-day rolling std", color="orange")
ax[1].axvline(split_date, color="black", linestyle="--")
ax[1].set_ylabel("Rolling Std")
ax[1].legend()

plt.tight_layout()
plt.show()
