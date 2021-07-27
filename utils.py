# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:54:53 2021

@author: pathouli
"""

def lda_fun(df_in, n_topics_in, num_words_in):
    import gensim
    import gensim.corpora as corpora
    #from gensim.models.coherencemodel import CoherenceModel
    
    data_tmp = df_in.str.split()
    id2word = corpora.Dictionary(data_tmp)
    
    corpus = [id2word.doc2bow(text) for text in data_tmp]

    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=n_topics_in, id2word=id2word, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=num_words_in)
    # print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  
    # coherence_model_lda = CoherenceModel(
    #     model=ldamodel, texts=data_tmp, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)
    for topic in topics:
        print(topic)
    return topics

def dictionary_check(var_in):
    import enchant
    d = enchant.Dict("en_US")
    tmp = var_in.split()
    tmp = [word for word in tmp if d.check(word)]
    tmp = ' '.join(tmp)
    return tmp

def rem_sw(var_in):
    import nltk
    from nltk.corpus import stopwords
    sw = stopwords.words('english')    
    clean_text = [word for word in var_in.split() if word not in sw]
    clean_text = ' '.join(clean_text)
    return clean_text

def unique_words(var_in):
    tmp = len(set(var_in.split()))
    return tmp

def open_pickle(path_in, file_name):
    import pickle
    tmp = pickle.load(open(path_in + file_name, "rb"))
    return tmp

def write_pickle(path_in, file_name, var_in):
    import pickle
    pickle.dump(var_in, open(path_in + file_name, "wb"))

def clean_text(var_in):
    import re
    tmp = re.sub("[^A-Za-z]+", " ", var_in.lower())
    return tmp

def seek_and_clean(path_in, path_out):
    import pandas as pd
    import os
    the_data = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
        tmp = root.split("/")[-1]
        for word in files:
            try:
                f = open(root + "/" + word, "r", encoding="utf8")
                tmp_txt = clean_text(f.read())
                if len(tmp_txt) > 0:
                    the_data = the_data.append(
                        {"body": tmp_txt, "label": tmp}, ignore_index=True)
                f .close()
            except:
                print ("ERROR WITH FILE: ", word)
                pass
    write_pickle(path_out, "data.pkl", the_data)
    return the_data

def token_cnt(var):
    tmp = len(set(var.split()))
    return tmp

def stem_fun(var):
    from nltk.stem import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var.split()]
    tmp = ' '.join(tmp)
    return tmp

def vec_fun(df_in, path_in, name_in):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    my_vec = CountVectorizer()
    
    my_vec_text = pd.DataFrame(my_vec.fit_transform(df_in).toarray())
    my_vec_text.columns = my_vec.get_feature_names()
    
    write_pickle(path_in + "output/", name_in + ".pkl", my_vec)
    return my_vec_text

def tf_idf_fun(df_in, path_in, name_in):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    my_tf_idf = TfidfVectorizer()
    my_tf_idf_text = pd.DataFrame(my_tf_idf.fit_transform(df_in).toarray())
    my_tf_idf_text.columns = my_tf_idf.get_feature_names()
        
    write_pickle(path_in + "output/", name_in + ".pkl", my_tf_idf)
    return my_tf_idf_text

def grid_search_fun(x_in, y_in, params_in, sw):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    if sw == "rf":
        my_rf = RandomForestClassifier(random_state=123)
    elif sw == "svm":
        my_rf = SVC(random_state=123)
    elif sw == "nb":
        my_rf = MultinomialNB()
    clf = GridSearchCV(my_rf, params_in)
    clf.fit(x_in, y_in)
    print ("Best Score:", clf.best_score_, "Best Params:", clf.best_params_)
    return clf.best_params_

def pca_fun(var, exp_var, path_o):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=exp_var)
    pca_data = pca.fit_transform(var)
    write_pickle(path_o, "pca.pkl", pca)
    print("# components:", len(pca.explained_variance_ratio_))
    print("explained variance:",sum(pca.explained_variance_ratio_))
    return pca_data

def perf_metrics(model_in, x_in, y_true):
    #How well did this model perform?
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = model_in.predict(x_in)
    metrics = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    return metrics

def my_rf(x_in, y_in, out_in, opt_param_in, sw):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    if sw == "rf":
        my_rf_m = RandomForestClassifier(**opt_param_in)
    elif sw == "svm":
        my_rf_m = SVC(**opt_param_in)
    elif sw == "nb":
        my_rf_m = MultinomialNB(**opt_param_in)
    my_rf_m.fit(x_in, y_in) #model is trained
    write_pickle(out_in, "rf.pkl", my_rf_m)
    return my_rf_m

def split_data(x_in, y_in, split_fraction):
    #training test split
    from sklearn.model_selection import train_test_split
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        x_in, y_in, test_size=split_fraction, random_state=42)
    return X_train_t, X_test_t, y_train_t, y_test_t

def my_cos_fun(df_in, xform_in, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    similarity = pd.DataFrame(cosine_similarity(df_in, xform_in))
    similarity.index = label_in
    return similarity

def my_pca(df_in, o_path):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    my_pca_txt = pca.fit_transform(df_in)
    write_pickle(o_path, "pca.pkl", pca)
    return my_pca_txt

def score_text(model_in, var_in):
    import numpy as np
    the_pred = model_in.predict(var_in)
    probs = model_in.predict_proba(var_in)
    print ("Predicted text:", the_pred[0], "With probability of:",
           str(round(np.max(probs)*100, 2)) + "%")
    return

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    from gensim.models import Word2Vec
    import pandas as pd
    from gensim.models import KeyedVectors
    import pickle
    my_model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    my_model = Word2Vec(df_in.str.split(),
                        min_count=1, vector_size=300)
    word_dict = my_model.wv.key_to_index
    #my_model.most_similar("calculus")
    #my_model.similarity("trout", "fish")
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(my_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pkl", "wb" ))
    return tmp_data, word_dict

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    wrd_dict = model.wv.key_to_index
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(base_path + "embeddings_domain.pkl")
    pickle.dump(model, open(path_in + "embeddings_domain_model.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pkl", "wb" ))
    return tmp_data, wrd_dict

def get_embed(path_in, file_name_in, var_in):
    import numpy as np
    embed_model = open_pickle(path_in, file_name_in)
    tmp = var_in.split()
    def get_score(var):
        try:
            tmp_arr = list()
            for word in tmp:
                tmp_arr.append(list(embed_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = get_score(var_in).reshape(1, -1)
    return tmp_out