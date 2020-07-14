import pandas as pd
import re
import nltk
nltk.download('punkt')
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

_wnl = nltk.WordNetLemmatizer()
def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def identity_features(ques):#,kh,kb):
    _id_words = ['is was are how can does which what type there']
    X = []
    for q in ques:
        clean_headline = clean(q)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _id_words]
        X.append(features)
    return X

def segg_test():

    dft=pd.read_csv("data/raw/train.csv",delimiter="|",header=None)
    dfs=pd.read_csv("data/raw/test.csv",delimiter="|",header=None)

    lt=dft.values.tolist()
    ls=dfs.values.tolist()

    vectorizer = TfidfVectorizer(max_features=500)

    qs,vs,yt,ys=[],[],[],[]

    for v in lt:
        qs.append(clean(v[1]))
        if v[2].lower()=='yes' or v[2].lower()=='no':
            yt.append('yn')
        else:
            yt.append('ot')
        
    for v in ls:
        vs.append(clean(v[1]))
        if v[2].lower()=='yes' or v[2].lower()=='no':
            ys.append('yn')
        else:
            ys.append('ot')
            
    Xt2= identity_features(qs)
    Xt1=vectorizer.fit_transform(qs).toarray()

    Xs2 = identity_features(vs)
    Xs1 = vectorizer.transform(vs).toarray()

    Xtrn=np.c_[Xt2,Xt1]
    Xtst=np.c_[Xs2,Xs1]

    clf=LinearSVC(loss='hinge',random_state=15,max_iter=16505)

    clf.fit(Xtrn, yt)

    pred=clf.predict(Xtst)

    return(pred)
