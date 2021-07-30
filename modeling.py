# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:50:37 2021

@author: pschw
"""

from crawler import *
from utils import * 

base_path = "C:/Users/pschw/Dropbox/Columbia/Courses/NLP for Python/Project/code/NLP_final_project/"

full_data = open_pickle(base_path, "full_data.pkl")

#data set on which we run our models on - remove all unnecessary columns
country_set = full_data.iloc[:,[1,2,3,6,7]]
country_set.columns = ["country","hdi_cat","hdi","sw_dict","stem"]


#convert text data in numerical data with vectorizer
my_vec_text = vec_fun(country_set.sw_dict, base_path, "vec")
my_vec_stem = vec_fun(country_set.stem, base_path, "vec_stem")

#convert text data in numerical data with tf-idf
my_tfidf_text = tf_idf_fun(country_set.sw_dict, base_path, "tf_idf")
my_tfidf_stem = tf_idf_fun(country_set.stem, base_path, "tf_idf_stem")

#apply pca analysis
pca_vec_text = my_pca(my_vec_text, base_path + "output/")
pca_vec_stem = my_pca(my_vec_stem, base_path + "output/")
pca_tfidf_text = my_pca(my_tfidf_text, base_path + "output/")
pca_tfidf_stem = my_pca(my_tfidf_stem, base_path + "output/")

# now we split the data in training and test set
# this is not for predictions themselves but to optimize the parameters
#x is the independent variable, y the dependent variable
x_train_param, x_pred, y_train_param_val, y_pred = split_data(
    my_tfidf_text, country_set.hdi_cat, 0.8)

#specify parameters
## random forrest
# parameters = {'n_estimators':[10, 100], 'max_depth':[None, 10, 100],
#               "random_state": [456]}
## support vector machine
# parameters = {'C':[0.01, 1.0], 'kernel':['linear', 'poly'],
#               "random_state": [123], 'probability': [True]}
## naive bayes
parameters = {'alpha':[0.001, 1.0], "fit_prior": [True, False]}

#chose model to use
flag = "nb"

#model training and specifications
optimal_params = grid_search_fun(x_pred, y_pred, parameters, flag)

#now we split into training and testing set
x_train, x_test, y_train, y_test = split_data(x_pred, y_pred, 0.2)

#train model with optimal parameters
rf_model = my_rf(
    x_train, y_train, base_path + "output/", optimal_params, flag)

#apply the random forrest on the test set
model_metrics = perf_metrics(rf_model, x_test, y_test)
