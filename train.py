from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from config import get_config
from utils import *
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import time
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd

def atomic_benchmark_estimator(estimator, X_test, verbose=False):
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return np.average(runtimes)

def train(config, X_train, y_train, n_heartrate):

    # from sklearn.datasets import load_iris
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import train_test_split

    # Calculate the mutual information of each feature with the target variable
    print("shape : ", X_train.shape)
    mi = mutual_info_classif(X_train, y_train)

    # Print the mutual information of each feature
    for x,info_gain in enumerate(mi):
        print(f"{x}: {info_gain}")

    # from sklearn.feature_selection import SelectKBest
    # from sklearn.feature_selection import mutual_info_classif

    # selector = SelectKBest(score_func=mutual_info_classif, k=50)

    # # Fit the selector to the data
    # selector.fit(X_train, y_train)

    # mask = selector.get_support()

    # for x, is_selected in enumerate(mask):
    #     if is_selected:
    #         print(x)
   
    columns=['Accuracy',	'Specificity',	'Sensitivity',	'F1 Score', 'param']
    no = ['']#,'','','','','','','','','','','']
    hr = np.empty(12)
    hr.fill(n_heartrate)
    hr = hr.tolist()
    hd = ['pantomkin']
    clf = ['adaboost']#,'catboost','adaboost']
    prm = ['tuned-iteration']#,'default','default']
    acc = []
    spe = []
    sns = []
    f1 = []
    rt = []
    parammodel=[]
    
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)   

    # ###========================================================= XGBOOST ===========================================================###

    params_xgboost={
         'n_estimators':[200,800],
         'max_depth':[10,15,20], 
         'objective':['reg:squarederror'], 
         'learning_rate': [0.03, 0.05, 0.1],
         'booster':['gbtree'],
         'min_child_weight':[1,3,5],
    }
   

    classifier = XGBClassifier()
    random_search= RandomizedSearchCV(classifier, param_distributions=params_xgboost,scoring=make_scorer(f1_score, average='weighted'),n_iter=10,n_jobs=-1,cv=cv,verbose=0)
    random_search.fit(X_train,y_train)
    best_param = random_search.best_params_
    parammodel.append(best_param)

    model1 = XGBClassifier(**best_param)
    # model1 = XGBClassifier()

    model_result1 = cross_validation(model1, X_train, y_train)

    for x in model_result1:
        print(x, model_result1[x])
    print()
    acc.append(model_result1['acc']/100)
    spe.append(model_result1['spe'])
    sns.append(model_result1['sns'])
    f1.append(model_result1['f1'])    

    # model1.fit(X_train,y_train)
    # atomic_runtimes = atomic_benchmark_estimator(model1, X_train, False)
    # print("Runtetime model1 avg = ",atomic_runtimes)
    # rt.append(atomic_runtimes)  

    # ###=============================================================================================================================###
    




    ###========================================================= CATBOOST ===========================================================###
    params_catboost={
         'iterations':[1200],
         'learning_rate': [0.15],
        #  'bootstrap_type':['Poisson','MVS', 'Bayesian','Bernoulli'],
        #  'bagging_temperature' : [0],
         'depth' : [14],
        #  'grow_policy' : ['SymmetricTree','Depthwise','Lossguide'],
         'verbose' : [True]
    }
    param={
         'iterations':1200,#600,800,1000,1200,1400],
         'learning_rate': 0.15,#0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24],
        #  'bootstrap_type':['Poisson','MVS', 'Bayesian','Bernoulli'],
        #  'bagging_temperature' : [0,1,2,3,4,5],
         'depth' : 10,
        #  'grow_policy' : ['SymmetricTree','Depthwise','Lossguide'],
    }
   

    # classifier = CatBoostClassifier()
    # random_search= RandomizedSearchCV(classifier, param_distributions=params_catboost,scoring=make_scorer(f1_score, average='weighted'),n_iter=1,n_jobs=-1,cv=cv,verbose=0)
    # random_search.fit(X_train,y_train)
    # best_param = random_search.best_params_
    # parammodel.append(best_param)
    # model2 = CatBoostClassifier(**best_param)

    parammodel.append(param)
    # model2 = CatBoostClassifier(**param)
    model2 = CatBoostClassifier()
    print("X_train[0]")
    print(X_train[0])
    print("y_train[0]")
    print(y_train[0])

    model_result2 = cross_validation(model2, X_train, y_train)

    for x in model_result2:
        print(x, model_result2[x])
    print()
    acc.append(model_result2['acc']/100)
    spe.append(model_result2['spe'])
    sns.append(model_result2['sns'])
    f1.append(model_result2['f1'])

    # model2 = CatBoostClassifier(**param)
    # model2.fit(X_train,y_train)

    # model2.save_model("catboostCCP",
    #        format="cpp",
    #        export_parameters=None,
    #        pool=None)


    # atomic_runtimes = atomic_benchmark_estimator(model2, X_train, False)
    # print("Runtetime model 2 avg = ",atomic_runtimes)
    # rt.append(atomic_runtimes)  

    ###=============================================================================================================================###

    ###========================================================= ADABOOST ===========================================================###
    params_adaboost = {
            'algorithm': ['SAMME', 'SAMME.R'],
            "n_estimators": [25,50,75,100],
            'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
    }

    classifier = AdaBoostClassifier()
    random_search= RandomizedSearchCV(classifier, param_distributions=params_adaboost,scoring=make_scorer(f1_score, average='weighted'),n_iter=10,n_jobs=-1,cv=cv,verbose=0)
    random_search.fit(X_train,y_train)
    best_param = random_search.best_params_
    parammodel.append(best_param)

    # model3 = AdaBoostClassifier(**best_param)
    model3 = AdaBoostClassifier()

    model_result3 = cross_validation(model3, X_train, y_train)

    for x in model_result3:
        print(x, model_result3[x])
    print()
    acc.append(model_result3['acc']/100)
    spe.append(model_result3['spe'])
    sns.append(model_result3['sns'])
    f1.append(model_result3['f1'])

    # model3.fit(X_train,y_train)
    # atomic_runtimes = atomic_benchmark_estimator(model3, X_train, False)
    # print("Runtetime avg model 3 = ",atomic_runtimes)
    # rt.append(atomic_runtimes)  

    # ###=============================================================================================================================###


    df = pd.DataFrame(list(zip(acc,spe,sns, f1,parammodel)), columns=columns)

    with pd.ExcelWriter('TA_model_perf_2.xlsx',mode='a',engine='openpyxl',if_sheet_exists='overlay') as writer:  
        df.to_excel(writer, sheet_name="Sheet1",header=None, startrow=writer.sheets["Sheet1"].max_row,index=False)
  
def cross_validation(model, _X, _y): 
      # def sensitivity(y_true,y_pred):
      #   cm=confusion_matrix(y_true, y_pred)
      #   FP = cm.sum(axis=0) - np.diag(cm)  
      #   FN = cm.sum(axis=1) - np.diag(cm)
      #   TP = np.diag(cm)
      #   TN = cm.sum() - (FP + FN + TP)
      #   Sensitivity = list()  
      #   for i in range(4):
      #    if (TP[i] == 0) and (FN[i]==0):
      #       Sensitivity.append(1)
      #    else:
      #       Sensitivity.append(TP[i]/(TP[i]+FN[i]))
      #   return np.mean(Sensitivity)

      def specificity(y_true,y_pred):
      
        cm=confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        Specificity = TN/(TN+FP)    
        return np.mean(Specificity)

      scoring = {
           'acc': 'accuracy',
         #   'prec_macro': 'precision_macro',
         #  'rec_macro': 'recall_macro',
         #   'prec_micro': 'precision_micro',
         #   'rec_micro': 'recall_micro',
           'sensitivity' : 'recall_macro',
           'specificity' : make_scorer(specificity) ,
           'f1_score' : make_scorer(f1_score, average='weighted')
           }

      cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)   
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=cv,
                               scoring=scoring,
                               return_train_score=True)
      return {
            #   "Training Accuracy scores": results['train_acc'],
            #   "Mean Training Accuracy": results['train_acc'].mean()*100,

            #   "Training Precision Micro scores": results['train_prec_micro'],
            #   "Mean Training Precision Micro": results['train_prec_micro'].mean(),

            #   "Training Precision Macro scores": results['train_prec_macro'],
            #   "Mean Training Precision Macro": results['train_prec_macro'].mean(),
              
            #   "Training Recall Micro scores": results['train_rec_micro'],
            #   "Mean Training Recall Micro": results['train_rec_micro'].mean(),

            #   "Training Recall Macro scores": results['train_rec_macro'],
            #   "Mean Training Recall Macro": results['train_rec_macro'].mean(),

            #   "Training F1 scores": results['train_f1_score'],
            #   "Mean Training F1 Score": results['train_f1_score'].mean(),

            #   "Validation Accuracy scores": results['test_acc'],
            #   "Mean Validation Accuracy": results['test_acc'].mean()*100,
                "acc": results['test_acc'].mean()*100,

            #   "Validation Precision Macro scores": results['test_prec_macro'],
            #   "Mean Validation Precision Macro": results['test_prec_macro'].mean(),

            #   "Validation Precision Micro scores": results['test_prec_micro'],
            #   "Mean Validation Precision Micro": results['test_prec_micro'].mean(),

            #   "Validation Recall Macro scores": results['test_rec_macro'],
            #   "Mean Validation Recall Macro": results['test_rec_macro'].mean(),

            #   "Validation Recall Micro scores": results['test_rec_micro'],
            #   "Mean Validation Recall Micro": results['test_rec_micro'].mean(),

            #   "Validation Specificity scores": results['test_specificity'],
            #   "Mean Validation Specificity": results['test_specificity'].mean(),
                "spe": results['test_specificity'].mean(),

            #   "Validation Specificity scores": results['test_specificity'],
            #   "Mean Validation Sensitivity": results['test_sensitivity'].mean(),
                "sns": results['test_sensitivity'].mean(),

            #   "Validation F1 scores": results['test_f1_score'],
            #   "Mean Validation F1 Score": results['test_f1_score'].mean()
                "f1": results['test_f1_score'].mean()
              }

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

def main(config):
    x_name, y_name, format = 'X-nonfull-rrOnly', 'y-nonfull-rrOnly', 'hr.hdf5'
    # x_name, y_name, format = 'X-turunan-featureSelect-', 'y-turunan-featureSelect-', 'hr.hdf5'
    # hr = [3,5,7,9]
    hr = [6]
    for n_hr in hr:
        print(n_hr , " - heart rate")
        (X,y) = loaddata(config.input_size, config.feature, x_name + str(n_hr) + format, y_name + str(n_hr) + format)
        # (X1,y1) = loaddata(config.input_size, config.feature,'X-turunan-plus-100-arr-6hr.hdf5', 'y-turunan-plus-100-arr-6hr.hdf5')
        print(X.shape)
        # print(X1.shape)
        # print(y.shape)
        # print(y1.shape)

        # X = np.concatenate((X,X1))
        # y = np.concatenate((y,y1))
        # print(X.shape)
        # print(X1.shape)
        unique2, counts2 = np.unique(y, return_counts=True)
        print(dict(zip(unique2, counts2)))
        def oversampling(X,y):
            from imblearn.over_sampling import SMOTE
            oversample = SMOTE(k_neighbors=4)
            X, y = oversample.fit_resample(X, y)
            return X, y

        unique2, counts2 = np.unique(y, return_counts=True)
        print("before",dict(zip(unique2, counts2)))
        X,y = oversampling(X, y)
        unique2, counts2 = np.unique(y, return_counts=True)
        print("after",dict(zip(unique2, counts2)))
        train(config, X, y, n_hr)

if __name__=="__main__":
    config = get_config()
    main(config)