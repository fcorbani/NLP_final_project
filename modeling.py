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
data_set = full_data.iloc[:,[1,2,3,6,7]]
data_set.columns = ["country","hdi_cat","hdi","sw_dict","stem"]


