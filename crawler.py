# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:05:36 2019

@author: pathouli
"""

def my_scraper(tmp_url_in):
    from bs4 import BeautifulSoup
    import requests
    import re
    tmp_text = ''
    try:
        content = requests.get(tmp_url_in, timeout=10)
        soup = BeautifulSoup(content.text, 'html.parser')

        tmp_text = soup.findAll('p') 

        tmp_text = [word.text for word in tmp_text]
        tmp_text = ' '.join(tmp_text)
        tmp_text = re.sub('\W+', ' ', re.sub('xa0', ' ', tmp_text))
    except requests.exceptions.Timeout:
        pass
    return tmp_text

def fetch_urls(query_tmp, cnt):
    #now lets use the following function that returns
    #URLs from an arbitrary regex crawl form google

    #pip install pyyaml ua-parser user-agents fake-useragent
    import requests
    from fake_useragent import UserAgent
    from bs4 import BeautifulSoup
    import re 
    ua = UserAgent()

    query = '+'.join(query_tmp.split())
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(cnt)
    #google_url = "https://www.google.com/search?q=" + query + "+news&source=lnms&tbm=nws" #works
    #google_url = "https://www.bing.com/search?q=" + query #+ "&num=" + str(cnt) #bing
    print (google_url)
    response = requests.get(google_url, {"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

    links = []
    titles = []
    descriptions = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()

            # Check to make sure everything is present before appending
            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
        # Next loop if one element is not present
        except:
            continue  

    to_remove = []
    clean_links = []
    for i, l in enumerate(links):
        clean = re.search('\/url\?q\=(.*)\&sa',l)

        # Anything that doesn't fit the above pattern will be removed
        if clean is None:
            to_remove.append(i)
            continue
        clean_links.append(clean.group(1))

    return clean_links

def clean_txt(var_in):
    '''
    a function that lower cases letters, removes special characters
    and stopwords, and performs an American dictionary check
    '''
    import enchant
    import re
    import nltk
    from nltk.corpus import stopwords
    #remove special characters and lower case letters
    var_in = re.sub("[^A-Za-z]+"," ",var_in.lower()) 
    sw = stopwords.words('english')    
    #remove stopwords
    clean_text = [word for word in var_in.split() if word not in sw]
    #do dictionary check
    d = enchant.Dict("en_US")
    clean_text = [word for word in clean_text if d.check(word)]
    clean_text = ' '.join(clean_text)
    return clean_text

def stem_txt(var_in):
    # to get the words stems of a text string
    from nltk.stem import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var_in.split()]
    tmp = ' '.join(tmp)
    return tmp
 
def write_crawl_results(my_query, the_cnt_in):
    #let use fetch_urls to get URLs then pass to the my_scraper function 
    import re
    import pandas as pd

    tmp_pd = pd.DataFrame()       
    for q_blah in my_query:
        the_urls_list = fetch_urls(q_blah, the_cnt_in)

        for word in the_urls_list:
            tmp_txt = my_scraper(word)
            #tmp_sw_txt = clean_txt(tmp_txt)
            #tmp_sw_stem_txt = stem_txt(tmp_sw_txt)
            if len(tmp_txt) != 0:
                try:
                    tmp_pd = tmp_pd.append({'body_basic': tmp_txt,
                                            #'body_sw': tmp_sw_txt,
                                            #'body_sw_stem': tmp_sw_stem_txt,
                                            'label': re.sub(' ', '_', q_blah)
                                            }, ignore_index=True)
                    print (word)
                except:
                    pass
    return tmp_pd


def word_freq(query,n_links=10):
    import pandas as pd
    #get body and labeled queries
    data = write_crawl_results(query, n_links) 
    # create empty dictionary
    my_dict = dict() 
    for name in set(data.label):
        #create empty dataframe
        my_pd = pd.DataFrame() 
        # collect all text bodies related to the query element in a series
        my_sr = data[data.label==name].body_basic 
        #merge all text bodies into string
        my_str = ' '.join(my_sr).lower() 
        # get a set of unique words in the string
        my_set = set(my_str.split()) 
        for word in my_set:
            #count how often each word appears in the string
            word_cnt = my_str.count(word) 
            #add unique words and their frequency to the df
            my_pd = my_pd.append({"Word":word, "Frequency":word_cnt},ignore_index=True) 
        #locate query names as keys and assign the dfs to respective values
        my_dict[name] = my_pd 
    return my_dict
