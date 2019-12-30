################################################################################
#This project deals with the analysis of 5 years of Amazon's closing stock     #
#prices registered in the period between 2013 and 2018.                        #
#Namely, we will try to train an ARIMA forecasting model and test its accuracy.#
#The dataset used is available at:                                             #
#https://www.kaggle.com/camnugent/sandp500                                     #
################################################################################

library(fpp2)
library(tseries)

source("import_dataset.R")
source("data_understanding.R")

# Dataset Import
dataset <- import_dataset("AMZN_data.csv")

# Data Understading
data_summary <- data_understanding(dataset)

# The dataset includes 1259 instances
# Each instance is made up of seven fields:
# - the date of the registration
# - the opening price (USD)
# - the highest price (USD)
# - the lowest price (USD)
# - the closing price (USD)
# - the volume shared (USD)
# - the Nasdaq index of the company (i.e. AMZN)
# There are no duplicated and no missing values

# Therefore, we only have to extract the "close" column of the dataset

# Embed the time series inside a ts object
# No particular seasonality is assumed
time_series <- ts(dataset$close, frequency = 1, start = 1)
# Display the time plot
autoplot(time_series) +
  ggtitle("Amazon daily changes in closing stock prices") +
  xlab("Day") + 
  ylab("Value (USD)")

# As usual, let'estimate the trend with a simple moving average
# For example, we may use a centred average of order 20...
trend_estimate_20 <- ma(time_series, order=20, centre = TRUE)
autoplot(time_series, series="Original") +
  autolayer(trend_estimate_20, series="20-MA") +
  xlab("Day") + ylab("Value(USD)") +
  ggtitle("Amazon daily changes in closing stock prices") +
  scale_colour_manual(values=c("Original"="grey","20-MA"="red"),
                      breaks=c("Original","20-MA"))
#...a centered average of order 100...
trend_estimate_100 <- ma(time_series, order=100, centre = TRUE)
autoplot(time_series, series="Original") +
  autolayer(trend_estimate_100, series="100-MA") +
  xlab("Day") + ylab("Value(USD)") +
  ggtitle("Amazon daily changes in closing stock prices") +
  scale_colour_manual(values=c("Original"="grey","100-MA"="red"),
                      breaks=c("Original","100-MA"))
#...or a centered average of order 200
trend_estimate_200 <- ma(time_series, order=200, centre = TRUE)
autoplot(time_series, series="Original") +
  autolayer(trend_estimate_200, series="200-MA") +
  xlab("Day") + ylab("Value(USD)") +
  ggtitle("Amazon daily changes in closing stock prices") +
  scale_colour_manual(values=c("Original"="grey","200-MA"="red"),
                      breaks=c("Original","200-MA"))
# Notice that a greater order results in a smoother estimate

# The presence of the trend suggests that the series is not-stationary
# We can check this fact making an Augmented-Dickey-Fuller test
adf.test(time_series)
# p-value is >= 0.99 so we cannot reject the null hypotesis of non-stationarity


