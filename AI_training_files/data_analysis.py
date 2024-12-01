import pandas as pd

df = pd.read_csv('geospatial_data.csv')

# print first 5 rows of the data
print(df.head())

# print data types and non-null counts
# this will give us information about which columns have categorical features and need to encoded 
print(df.info())

print("++++++++++++++++++++++  Describe data +++++++++++++++++++++++")
print(df.describe())