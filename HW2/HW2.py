# OEAS805 HOMEWORK 2: EDA AND ENVIRONMENTS
#import libraries
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

#%% 
#load data
filepath ='C:/Users/noahr/Downloads/Important or Academic Files/Academic/Course Materials/Fall 2026/Adv Env Data Sci/OEAS805/data/International_Coastal_Cleanup.csv'
data = pd.read_csv(filepath, sep = ',')

#%% Size of your data set. How many variables does it contain? How many samples/data points (e.g. rows of data)?

data.shape #77 rows, 53 columns
data.size #4081 total elements

col = data.columns
col.size #number of variables (columns) = 53

### The size of the dataset is 77 rows x 53 columns (4081 elements total). There are 53 variables and 77 samples/datapoints in the dataset

#%% Data types in your data set. Are all of the variables the same type? Do you have a combination of types?

data.info() #float64(3), int64(39), str(11)

### There is a combination of strings and numerical values in this dataset. In total there are 11 string variables, 3 floats, and 39 integers

#%%Do your data columns contain missing data? Do all variables have the same number of missing data points?

any_nans = data.isna()
has_nans = any_nans.any().any()

### There are no NaN (missing) values in the dataset

#%%Descriptive statistics for your variables of (mean, stdev, median, etc...). 
#Boxplots, bar charts and histograms are great for visualizing the distribution of your data.

#make a copy of 'data' in which strings of numbers are converted to actual numbers for statistics
trash_vals = data.drop(columns=['Area', 'Date'])
trash_vals = trash_vals.replace(',', '', regex=True) #learned from python documentation
trash_vals = trash_vals.apply(pd.to_numeric, errors='coerce') #learned from python documentation

stats = trash_vals.describe() #descriptive stats

#boxplot of the mean amount of each trash item 
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=trash_vals,orient="h",)
plt.tight_layout()
plt.show()

#first boxplot was too squeezed. Use subplots with different axis scales
small_cols = trash_vals.columns[trash_vals.max() < 100]
med_cols = trash_vals.columns[(trash_vals.max() >= 100) & (trash_vals.max() <= 1000)]
large_cols = trash_vals.columns[trash_vals.max() > 1000]

fig, ax = plt.subplots(3,1)
sns.boxplot(data=trash_vals[small_cols], orient="h", ax=ax[0])
sns.boxplot(data=trash_vals[med_cols], orient="h", ax=ax[1])
sns.boxplot(data=trash_vals[large_cols], orient="h", ax=ax[2])
plt.tight_layout()
plt.show()

#bar graph of amount of trash per area
trash_per_area = pd.DataFrame({
    'Area': data['Area'],
    'Total Items Collected': trash_vals['Total Items Collected']
})
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=trash_per_area, x='Total Items Collected', y='Area',errorbar=None)
ax.set_xlabel("Frequency")
ax.set_ylabel(" ")
plt.tight_layout()
plt.show()

## 


#%%Does your data contain outliers?

### Yes, based on the box plots, most cateogories of the items collected have outliers

#%%How do your variables relate to each other? You might look at correlations, scatter plots and/or correlation matrices.

#correlate different types of trash with each other
cor = data.corr(numeric_only = True)
plt.figure(figsize=(18, 15))
sns.heatmap(cor,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    center=0,
    xticklabels=True,
    yticklabels=True)
plt.show()

#relationship between man power and total trash at each location
trash_manpower = pd.DataFrame({
    'Area': data['Area'],
    'Total Items Collected': trash_vals['Total Items Collected'],
    'Number of Volunteers': data['Number of Volunteers'],
    'Volunteer Hours': data['Volunteer Hours'],
})
trash_manpower_corr = trash_manpower.corr(numeric_only = True)

### In terms of number of a given item collect at each site, throwaway (paper, 
### plastic) tableware items were highly correlated with each other (i.e. where 
### one tableware item was found, other types were likely to be found)

### "Clean swell" items (e.g. "Fishing gear", "Other packaging", "Other trash")
### were not strongly correlated with any other type of item expect other swell
### items

### The total number of items collected at each site is highly correlated with 
### the number of volunteers (0.73) and the number of volunteer hours (0.80)


#%%What initial conclusions can you draw from your exploration of the data?

### The dataset includes counts of various items picked up during clean-up 
### efforts at diffferent locations

### The number of items picked up at any given site varies greatly (ranging 
### from 0 to >20,000)

### Certain types of items (such as tableware) are highly correlated with 
### each other

### The amount of items picked up at any given site is highly correlated with 
### the number of volunteers present and the duration of time the volunteers workd

