import numpy as np
from keras import *
from keras.preprocessing import *

maxlen = 9

def get_tokenizer_que(list_all,train_que,test_que):

    tokenizer_ = text.Tokenizer(num_words=1000)
    tokenizer_.fit_on_texts(list_all)
    wrdidx = tokenizer_.word_index

    train_que = tokenizer_.texts_to_sequences(train_que)
    train_que = sequence.pad_sequences(train_que, maxlen, padding='post', value=0, truncating='post')

    test_que = tokenizer_.texts_to_sequences(test_que)
    test_que = sequence.pad_sequences(test_que, maxlen, padding='post', value=0, truncating='post')

    return(train_que,test_que,wrdidx)

def get_tokenizer_ans_yn(train_ans):

    tokenizer_ = text.Tokenizer()
    tokenizer_.fit_on_texts(train_ans)
    train_ans = tokenizer_.texts_to_sequences(train_ans)
    train_hot = utils.to_categorical(train_ans, 3)  #one-hot

    return(train_hot)

def get_tokenizer_ans_ot(train_ans):

    ans_words = ''
    for lines in train_ans:
        ans_words += ' ' + lines

    dic_ = len(set(ans_words.split())) + 1

    tokenizer_ = text.Tokenizer(num_words=dic_)
    tokenizer_.fit_on_texts(train_ans)
    train_ans = tokenizer_.texts_to_sequences(train_ans)
    train_ans = sequence.pad_sequences(train_ans, maxlen, padding='post', value=0, truncating='post')
    train_hot = utils.to_categorical(train_ans, dic_)  #one-hot

    return(train_hot, dic_, tokenizer_)

def get_weight_matrixx(wrdidx):

    dic_emb1 = {}
    file1 = open('data/external/glove/glove.6B.300d.txt')
    for line in file1:
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        word_weights = np.asarray(values[1:], dtype=np.float32)
        dic_emb1[word] = word_weights

    dic_emb = {}
    file1 = open('data/external/glove/w2v.6B.300d.txt')
    for line in file1:
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        word_weights = np.asarray(values[1:], dtype=np.float32)
        if word in dic_emb1:     
            dic_emb[word] = np.concatenate([dic_emb1[word],word_weights])  
        else:     
            dic_emb[word] = np.asarray([0.0]* 600, dtype=np.float32)


    #GloVe

    len_wrdidx=len(wrdidx)+1
    embwrd=[]

    with open('data/external/glove/w2v.6B.300d.txt', 'r') as file: 
        for line in file:
            values = line.split() # Word and weights separated by space
            word = values[0] # Word is first symbol on each line
            if word in wrdidx:
                embwrd.append(word)
                
    file.close()

    weight_matrix2 = np.zeros((len_wrdidx,600))
    for word in dic_emb:
        if word in wrdidx:
            word_weights2 = dic_emb[word]
            index = wrdidx.get(word)
            weight_matrix2[index] = word_weights2

    for i in wrdidx:
        if i not in embwrd:
            weight_matrix2[wrdidx.get(i)]=np.asarray([0.0]* 600, dtype=np.float32)

    weight_matrixx = np.asarray(weight_matrix2)

    return(weight_matrixx)

def read_files(file_name):

    list_ = []
    f = open(file_name)
    lines = f.readlines()

    for line in lines:
        line = line.strip()
        list_.append(line)
    f.close()

    return(list_)

def get_train_que_yn():
    return(read_files('data/processed/train_yn.que'))

def get_train_que_ot():
    return(read_files('data/processed/train_ot.que'))

def get_test_que_yn():
    return(read_files('data/processed/test_yn.que'))

def get_test_que_ot():
    return(read_files('data/processed/test_ot.que'))

def get_train_ans_yn():
    return(read_files('data/processed/train_yn.ans'))
    
def get_train_ans_ot():
    return(read_files('data/processed/train_ot.ans'))   

def get_que_features_yn():

    train_que_yn = get_train_que_yn()
    test_que_yn = get_test_que_yn()

    list_que_yn = train_que_yn + test_que_yn
    train_que_yn, test_que_yn, wrdidx_yn = get_tokenizer_que(list_que_yn,train_que_yn,test_que_yn)

    weight_matrixx_yn = get_weight_matrixx(wrdidx_yn)

    return(train_que_yn, test_que_yn, weight_matrixx_yn, wrdidx_yn)

def get_que_features_ot():

    train_que_ot = get_train_que_ot()
    test_que_ot = get_test_que_ot()

    list_que_ot = train_que_ot + test_que_ot
    train_que_ot, test_que_ot, wrdidx_ot = get_tokenizer_que(list_que_ot,train_que_ot,test_que_ot)

    weight_matrixx_ot = get_weight_matrixx(wrdidx_ot)

    return(train_que_ot, test_que_ot, weight_matrixx_ot, wrdidx_ot)

def get_ans_features():

    train_ans_yn = get_train_ans_yn()
    train_ans_ot = get_train_ans_ot()

    train_hot_yn = get_tokenizer_ans_yn(train_ans_yn)
    train_hot_ot, ans_dic_ot, tokenizer_ = get_tokenizer_ans_ot(train_ans_ot)

    return(train_hot_yn,train_hot_ot, ans_dic_ot, tokenizer_)  