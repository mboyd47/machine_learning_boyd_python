#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
#%% import data
data = pd.read_csv("/Users/marissaboyd/Downloads/hospital_mortality.csv")
print(pd.crosstab(index = data['outcome'], columns = 'count'))
data = data[data['outcome'].notna()]
# %%
#check for missing values
missing = pd.DataFrame(data.isna().sum())
missing.reset_index(inplace=True)
#print(missing)
#%%
#relplacing missings with medians
data['BMI'].fillna(data['BMI'].median(), inplace = True)
data['heart rate'].fillna(data['heart rate'].median(), inplace = True)
data['Systolic blood pressure'].fillna(data['Systolic blood pressure'].median(), inplace = True)
data['Diastolic blood pressure'].fillna(data['Diastolic blood pressure'].median(), inplace = True)
data['Respiratory rate'].fillna(data['Respiratory rate'].median(), inplace = True)
data['temperature'].fillna(data['temperature'].median(), inplace = True)
data['SP O2'].fillna(data['SP O2'].median(), inplace = True)
data['Urine output'].fillna(data['Urine output'].median(), inplace = True)
data['Neutrophils'].fillna(data['Neutrophils'].median(), inplace = True)
data['Basophils'].fillna(data['Basophils'].median(), inplace = True)
data['Lymphocyte'].fillna(data['Lymphocyte'].median(), inplace = True)
data['PT'].fillna(data['PT'].median(), inplace = True)
data['INR'].fillna(data['INR'].median(), inplace = True)
data['Creatine kinase'].fillna(data['Creatine kinase'].median(), inplace = True)
data['glucose'].fillna(data['glucose'].median(), inplace = True)
data['Blood calcium'].fillna(data['Blood calcium'].median(), inplace = True)
data['PH'].fillna(data['PH'].median(), inplace = True)
data['Lactic acid'].fillna(data['Lactic acid'].median(), inplace = True)
data['PCO2'].fillna(data['PCO2'].median(), inplace = True)
# %%
#reseting index
data = data.reset_index(drop = True)
#%% 
#Data Exploration
data.hist(column = 'age', by = 'outcome')

data.hist(column = 'BMI', by = 'outcome')

data.hist(column = 'heart rate', by = 'outcome')

data.hist(column = 'Systolic blood pressure', by = 'outcome')

data.hist(column = 'Diastolic blood pressure', by = 'outcome')

data.hist(column = 'Respiratory rate', by = 'outcome')

data.hist(column = 'temperature', by = 'outcome')

data.hist(column = 'SP O2', by = 'outcome')

data.hist(column = 'Urine output', by = 'outcome')

data.hist(column = 'MCH', by = 'outcome')

data.hist(column = 'MCHC', by = 'outcome')

data.hist(column = 'MCV', by = 'outcome')

data.hist(column = 'RDW', by = 'outcome')

data.hist(column = 'Leucocyte', by = 'outcome')

data.hist(column = 'Platelets', by = 'outcome')

data.hist(column = 'Neutrophils', by = 'outcome')

data.hist(column = 'Basophils', by = 'outcome')

data.hist(column = 'Lymphocyte', by = 'outcome')

data.hist(column = 'PT', by = 'outcome')

data.hist(column = 'INR', by = 'outcome')

data.hist(column = 'NT-proBNP', by = 'outcome')

data.hist(column = 'Creatine kinase', by = 'outcome')

data.hist(column = 'Creatinine', by = 'outcome')

data.hist(column = 'Urea nitrogen', by = 'outcome')

data.hist(column = 'glucose', by = 'outcome')

data.hist(column = 'Blood potassium', by = 'outcome')

data.hist(column = 'Blood sodium', by = 'outcome')

data.hist(column = 'Blood calcium', by = 'outcome')

data.hist(column = 'Chloride', by = 'outcome')

data.hist(column = 'Anion gap', by = 'outcome')

data.hist(column = 'Magnesium ion', by = 'outcome')

data.hist(column = 'PH', by = 'outcome')

data.hist(column = 'Bicarbonate', by = 'outcome')

data.hist(column = 'Lactic acid', by = 'outcome')

data.hist(column = 'PCO2', by = 'outcome')

data.hist(column = 'EF', by = 'outcome')

# %%
#creating feature and target datasets
X = pd.DataFrame(data.drop(['outcome','ID','group','deficiencyanemias','hematocrit',
                            'RBC','Basophils','Creatine kinase'], axis = 1))
y = pd.array(data['outcome'])
# %%
#normalizing non-binary features
from sklearn.preprocessing import StandardScaler
x = X.drop(['gendera','hypertensive','atrialfibrillation',
            'CHD with no MI','diabetes','depression',
            'Hyperlipemia','Renal failure','COPD'], axis = 1)
x_scaled = StandardScaler().fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns = x.columns)
# %%
#adding binary features to new scaled feature data set
x_scaled['gender'] = X['gendera']
x_scaled['hypertensive'] = X['hypertensive']
x_scaled['atrialfibrillation'] = X['atrialfibrillation']
x_scaled['CHD with no MI'] = X['CHD with no MI']
x_scaled['diabetes'] = X['diabetes']
x_scaled['depression'] = X['depression']
x_scaled['Hyperlipemia'] = X['Hyperlipemia']
x_scaled['Renal failure'] = X['Renal failure']
x_scaled['COPD'] = X['COPD']
# %%
#dropping variables that won't be used for models
x_scaled2 = x_scaled.drop(['age','BMI','RDW','Leucocyte','Neutrophils',
                            'Lymphocyte','PT','INR','NT-proBNP',
                            'Urea nitrogen','PH','Bicarbonate','EF'],
                         axis = 1)
#%%
#correlation heatmap to help create engineered features
plt.figure(figsize = (25,16))
sns.heatmap(x_scaled2.corr(), annot = True)
# %%
## Feature Engineering ##
x_scaled2['mch_mcv'] = x_scaled2['MCH']*x_scaled2['MCV']
x_scaled2['mchc_mch'] = x_scaled2['MCH']* x_scaled2['MCHC']
x_scaled2['BloodSodium_chloride'] = x_scaled2['Blood sodium']*x_scaled2['Chloride']
x_scaled2['aniongap_creatinine'] = x_scaled2['Anion gap']*x_scaled2['Creatinine']
x_scaled2['heart_2'] = x_scaled2['heart rate']*x_scaled2['heart rate']
# %%
x_scaled2.to_pickle('features.pkl')
y = pd.DataFrame(y, columns = ['outcome'])
# %%
y.to_pickle('target.pkl')
# %%
