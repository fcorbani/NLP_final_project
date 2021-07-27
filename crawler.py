# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:05:36 2019

@author: pathouli
"""

def init():
    from torpy import TorClient
    hostname = 'ifconfig.me'  # It's possible use onion hostname here as well
    with TorClient() as tor:
        # Choose random guard node and create 3-hops circuit
        with tor.create_circuit(3) as circuit:
            # Create tor stream to host
            with circuit.create_stream((hostname, 80)) as stream:
                # Now we can communicate with host
                stream.send(b'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % hostname.encode())
                recv = stream.recv(1024)
    return 0

def my_scraper(tmp_url_in):
    from bs4 import BeautifulSoup
    import requests
    import re
    import time
    tmp_text = ''
    try:
        content = requests.get(tmp_url_in, timeout=10)
        soup = BeautifulSoup(content.text, 'html.parser')

        tmp_text = soup.findAll('p') 

        tmp_text = [word.text for word in tmp_text]
        tmp_text = ' '.join(tmp_text)
        tmp_text = re.sub('\W+', ' ', re.sub('xa0', ' ', tmp_text))
    except:
        print("Connection refused by the server..")
        print("Let me sleep for 5 seconds")
        print("ZZzzzz...")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        pass
    # except requests.exceptions.Timeout:
    #     pass
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
 
def write_crawl_results(my_query, the_cnt_in):
    #let use fetch_urls to get URLs then pass to the my_scraper function 
    import re
    import pandas as pd

    tmp_pd = pd.DataFrame()       
    for q_blah in my_query:
        init()
        the_urls_list = fetch_urls(q_blah, the_cnt_in)

        for word in the_urls_list:
            tmp_txt = my_scraper(word)
            if len(tmp_txt) != 0:
                try:
                    tmp_pd = tmp_pd.append({'body_basic': tmp_txt,
                                            'label': re.sub(' ', '_', q_blah)
                                            }, ignore_index=True)
                    print (word)
                except:
                    pass
    return tmp_pd