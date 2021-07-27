#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:09:08 2021

@author: faustinecorbani
"""

#Make sure you also have the crawler.py and utils.py files from class open
from crawler import *
from utils import * 

#Import the libraries we'll us
import pandas as pd
import re 

#Download the HDI csv files to your computer using the following link:
#http://hdr.undp.org/en/indicators/137506?fbclid=IwAR3rd-tPpSq3YLzgeQjK6JPoTzyVof67bNbN2MxWwFEYUfuACSDeuhHY00E#

base_path = "/Users/faustinecorbani/Desktop/SCAN Lab/Natural Language Processing/final_project" #path to the HDI datafile

#Step 1: Read and clean the hdi_data

#The first 5 and last 18 rows of the df are useless text which is why we use the skiprows and skipfooter argument
hdi_data = pd.read_csv(base_path + "/Human Development Index (HDI).csv", skiprows = 5, skipfooter=18, index_col = False) 

#Keep only the columns we want
hdi_data = hdi_data[['Country','2019']]

#Clean the Country column
hdi_data['Country'] = hdi_data['Country'].str.strip() 

#Create a function that assigns a rank of very high, high, medium, or low
# based on a country's HDI score

def hdi_score_to_rank(score):
    if score >= 0.8:
        return "Very high"
    if score > 0.7 and score < 0.7999:
        return "High"
    if score > 0.550 and score < 0.6999:
        return "Medium"
    if score <= 0.549:
        return "Low"

#apply this function to our data
hdi_data["Rank"] = hdi_data["2019"].apply(lambda x: hdi_score_to_rank(x))

#Uncomment this code to drop the "2019" column
#hdi_data = hdi_data.drop('2019', axis=1)  


#Step 2: Use the crawler to create a dataframe containing one row for each query per Country

#Create our query
list_of_countries = hdi_data['Country'].to_list() #this will be our query
test_countries = ["France","Ukraine", "Zimbabwe","China"] #this runs faster than using the full list of countries and can be helpful for testing things out

#Run our crawler
text_data = write_crawl_results(list_of_countries, 3) #use test_countries instead if needed


#Step 3: Merge the hd_data from our crawler to our initial dataframe 

#Rename label column to Country column to use it to merge
text_data = text_data.rename(columns={"label" : "Country"})

#Merge both dfs to create a df with one row per query
full_data_long = pd.merge(the_data,hdi_data, on='Country')

#Use this code if you want one row per country instead (i.e. it aggregates all the queries)
full_data_long["body_basic"] = full_data_long["body_basic"].apply(lambda x: x.ljust(len(x)+1)) #ads a space at the end of each string so that when we aggregate them words won't stick to one another 
aggregation_functions = {'body_basic': 'sum', 'Country': 'first', "Rank": "first", "2019" : "first"} # Remove ' "2019": "first" ' if you dropped that column earlier
full_data_short = full_data_long.groupby(full_data_long['Country']).aggregate(aggregation_functions)


#Let's save the first versions of our cleaned data before we edit them more
write_pickle(base_path, "hdi_data_long.pkl", full_data_long)
write_pickle(base_path, "hdi_data_short.pkl", full_data_short)

 
#Step 4: Create new versions our body_basic column with stemming, using the english dictionnary etc..
full_data = open_pickle(base_path, "hdi_data_short.pkl") #open whichever version you'd like to use for the analysis

#Clean the body of text by removing anything besides characters
full_data["body_clean"] = full_data.body_basic.apply(clean_text) 

#Remove stop words
full_data["rem_sw"] = full_data.body_clean.apply(rem_sw) 

#Dictionnary check
full_data["sw_dictionary_check"] = full_data.rem_sw.apply(dictionary_check) 

#Stemming
full_data["stem_dict_sw"] = full_data.sw_dictionary_check.apply(stem_fun)

#Use .apply(token_cnt) if interested in the length of the resulting body of text
