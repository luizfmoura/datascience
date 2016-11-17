# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:28:10 2016

@author: 14036281739
"""

def read_data(pathName, fileName):
    import pandas as pd
    import os   
    
    filePath = os.path.join(pathName,fileName)
    return pd.read_csv(filePath)

    
def create_map():
    ## List of tuples with name and number of repititons.
    name_list = [('infections', 139),
                ('neoplasms', (239 - 139)),
                ('endocrine', (279 - 239)),
                ('blood', (289 - 279)),
                ('mental', (319 - 289)),
                ('nervous', (359 - 319)),
                ('sense', (389 - 359)),
                ('circulatory', (459-389)),
                ('respiratory', (519-459)),
                ('digestive', (579 - 519)),
                ('genitourinary', (629 - 579)),
                ('pregnancy', (679 - 629)),
                ('skin', (709 - 679)),
                ('musculoskeletal', (739 - 709)),
                ('congenital', (759 - 739)),
                ('perinatal', (779 - 759)),
                ('ill-defined', (799 - 779)),
                ('injury', (999 - 799))]
    ## Loop over the tuples to create a dictionary to map codes 
    ## to the names.
    out_dict = {}
    count = 1
    for name, num in name_list:
        for i in range(num):
          out_dict.update({str(count): name})  
          count += 1
    return out_dict
  

def map_codes(df, codes):
    import pandas as pd
    col_names = df.columns.tolist()
    for col in col_names:
        temp = [] 
        for num in df[col]:           
            if ((num is None) | (num in ['unknown', '?', 'Unknown']) | (pd.isnull(num))): temp.append('unknown')
            elif(num.upper()[0] == 'V'): temp.append('supplemental')
            elif(num.upper()[0] == 'E'): temp.append('injury')
            else: 
                lkup = num.split('.')[0]
                temp.append(codes[lkup])
        df.loc[:, col] = temp
        #df[col] = temp
    return df

    
def prep_data(pathName = '.', fileName = 'diabetic_data.csv', admissionFileName = 'admissions_mapping.csv'):
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.cross_validation import train_test_split,cross_val_score

    from sklearn.svm import LinearSVC    
    
    diabetics_data = read_data(pathName,fileName)
    admissions_mapping = read_data(pathName,admissionFileName)
    
    admissions_mapping['admission_type_description'] = ['Unknown' if ((x in ['Not Available', 'Not Mapped', 'NULL']) | (pd.isnull(x))) else x 
                                                 for x in admissions_mapping['admission_type_description']]

 
    df = pd.merge(diabetics_data,admissions_mapping,on="admission_type_id",how="inner")
    
    df.drop(['admission_type_id','encounter_id','patient_nbr'], axis = 1, inplace = True)
    
    df.dropna(axis = 0, how='all', inplace = True)    
    
    df.drop_duplicates(inplace=True)
    
    #Clean Missing Data
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_columns:
        df[col].replace('?','unknown',inplace=True)
        
    col_list = ['diag_1', 'diag_2', 'diag_3']
    codes = create_map()
    df[col_list] = map_codes(df[col_list], codes)
        
    df['readmitted'] = ['NO' if (y == 'NO') else 'YES' for y in df['readmitted']]
        
    scaled_cols = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses']
    df[scaled_cols] = preprocessing.scale(df[scaled_cols])    
                        
    dic_features = df[df.columns.difference(['readmitted'])].to_dict('records')    
    
    vec = DictVectorizer()
                        
    vec_features = vec.fit_transform(dic_features).toarray()
    
    vec_labels = np.array([1 if x == 'YES' else 0 for x in df['readmitted']])
    
    return vec_features,vec_labels

def split_data(vec_features, vec_labels):
    from sklearn.cross_validation import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(vec_features, vec_labels,test_size=0.5,random_state=0)
    
    return X_train, X_test, y_train, y_test
    
def train(X_train,y_train):    
    from sklearn.linear_model import LinearRegression    

    estimator = LinearRegression()    
    estimator.fit(X_train,y_train)
    
    return estimator
    
def evaluate(estimator,X_test,y_test):
    import numpy as np    
    import sklearn.metrics as met    
    from sklearn.metrics import confusion_matrix
    from sklearn.cross_validation import cross_val_score
    
    predict_probs = estimator.predict(X_test)
    predict_labels = np.array([1 if x > 0.5 else 0 for x in predict_probs])    
    
    conf_mat = confusion_matrix(y_test,predict_labels)
    acc_score = met.accuracy_score(y_test,predict_labels)
    f1_score = met.f1_score(y_test,predict_labels)
    rec_score = met.recall_score(y_test,predict_labels)
    prec_score = met.precision_score(y_test,predict_labels)
    
    print('True Positive: {0:d}'.format(conf_mat[0,0]))
    print('False Negative: {0:d}'.format(conf_mat[0,1]))
    print('False Positive: {0:d}'.format(conf_mat[1,0]))
    print('True Negative: {0:d}'.format(conf_mat[1,1]))
    
    print('Accuracy: {0:.2f}'.format(acc_score))
    print('F1: {0:.2f}'.format(f1_score))
    print('Recall: {0:.2f}'.format(rec_score))
    print('Precision: {0:.2f}'.format(prec_score))
    
def main():
#import os
#os.chdir("C:/Users/14036281739/Documents/Conhecimento/data_sciente_ms/Course 7 - Principles of Machine Learning")
#os.chdir("C:/Users/Luiz/Documents/workspace/Data Science Curriculum from Microsoft/Course 7 - Principles of Machine Learning")
#import diabetes_classification as dc
#dc.main()   
    vec_features,vec_labels = prep_data()    
    X_train, X_test, y_train, y_test = split_data(vec_features,vec_labels)    
    estimator = train(X_train,y_train)    
    evaluate(estimator,X_test,y_test)
    