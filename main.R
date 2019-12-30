################################################################################
#This project deals with the analysis of 5 years of Amazon's closing stock     #
#prices registered in the period between 2013 and 2018.                        #
#Namely, we will try to train an ARIMA forecasting model and test its accuracy.#
#The dataset used is available at:                                             #
#https://www.kaggle.com/camnugent/sandp500                                     #
################################################################################

source("import_dataset.R")
source("data_understanding.R")

# Dataset Import
dataset <- import_dataset("AMZN_data.csv")

# Data Understading
data_summary <- data_understanding(dataset)

# The dataset includes 1259 instances
# Each instance is made up of seven fields:
# - the date of the registration
# - the opening price
# - the highest price
# - the lowest price
# - the closing price
# - the volume shared
# - the Nasdaq index of the company (i.e. AMZN)
# There are no duplicated and no missing values

# Therefore, we only have to etract the "close" column of the dataset


