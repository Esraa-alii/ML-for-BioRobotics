import pandas as pd
import numpy as np
def delete_missingData (data):
    #remove rows of missing data
    data=pd.DataFrame(data)
    data.dropna(inplace=True)
    return data

def scratch_fillMissing_Value(data,col,value):
    #fill the missing data with specific value
    data = pd.DataFrame(data)
    data[col].fillna(value = value, inplace=True)
    return data

def scratch_fillMissing(data,col,strategy):
    #fill the missing data with specific strategy
    data = pd.DataFrame(data)
    if(strategy == 'mean'):
        data[col].fillna(value =data[col].mean(), inplace=True)
    elif(strategy == 'median'):
        data[col].fillna(value =data[col].median(), inplace=True)   
    elif(strategy == 'max'):
        data[col].fillna(value =data[col].max(), inplace=True) 
    elif(strategy == 'min'):
        data[col].fillna(value =data[col].min(), inplace=True)    
  
        
    return data 

#example for cleanup list
cleanup_data = {1:     {"male": 1, "female": 0},
                4: {"yes": 1, "no": 0},
                5: {"southeast":1, "northwest":0,"southwest":2,"northeast":3}}

def scratch_manual_encoding(data,cleanup_data):
    data = data.replace(cleanup_data) 
    return data 

def scratch_labelEncoding(data,col):
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes
    return data


def scratch_oneHotEncoding(data,col):
    encoded_data=pd.get_dummies(data[col])
    data = pd.concat([data, encoded_data], axis=1)
    data = data.drop(col,axis=1)
    return data

def scratch_Standardization(X):
    #Standardization in range of -3:3
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

def scratch_normalization_minMax(X):
    #normalization in range of -1:1
    Xmax = np.max(X,axis=0)
    Xmin = np.min(X,axis=0)
    X_norm = (X - Xmin) / (Xmax-Xmin)
    return X_norm

def detect_outliers(data,threshold):
    mean = np.mean(data)
    std =np.std(data)
    outliers=[]
    for i in data:
        z_score= (float(i) - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def detect_outlier_Z_score(data,threshold):
    mean = np.mean(data)
    std =np.std(data)
    outliers=[]
    for i in data:
        z_score= (i - mean)/std 
        if abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def detect_outliers_IQR(data,lower_bound_val,upper_bound_val):
    outliers=[]
    for i in data:
        if i < lower_bound_val or i >upper_bound_val :
            outliers.append(i)
    return outliers


def drop_outliers(data,field_name):
    iqr = 1.5 * (np.percentile(data[field_name] , 75) - np.percentile(data[field_name],25))
    data.drop(data[data[field_name] > (iqr + np.percentile(data[field_name],75))].index , inplace=True)
    data.drop(data[data[field_name] < (np.percentile(data[field_name],25)-iqr)].index , inplace=True)

def outliers_imputation(data,field_name,strategy):
    for i in data[field_name]:
        q1 = data[field_name].quantile(0.25)
        q3 = data[field_name].quantile(0.75)
        iqr = q3 - q1
        lower_tail = q1 - 1.5 * iqr
        upper_tail = q3 + 1.5 * iqr
        if i > upper_tail or i < lower_tail:
            if strategy == 'mean':
                data[field_name] = data[field_name].replace(i,np.mean(data[field_name]))
            elif strategy == 'median':
                data[field_name] = data[field_name].replace(i,np.median(data[field_name]))
            elif strategy == 'zero':
                data[field_name] = data[field_name].replace(i,0)
    