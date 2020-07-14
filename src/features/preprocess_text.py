import nltk
import re

nltk.download('punkt', quiet=True)
lemma = nltk.WordNetLemmatizer()

def pretreat_que():

    data_list= ['train','test']

    for data in data_list:
        
        fw = open('data/processed/' + data + '.que', 'w')
        f = open('data/intermediate/' + data + '.q')
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '').split(' ') 
            j = len(line)
            for i in range(j):
                for div in ['the','of','in','and','a','with','to','an','at','on','from','after','into','was','does']:  
                    if line[i] == div:
                        line[i] = ''
                if re.match(r'[0-9]+', line[i]): 
                    line[i] = 'num'
                elif re.match(r'[a-z][0-9]+', line[i]):
                    line[i] = 'pos'
                if line[i] != '':
                    line[i] = lemma.lemmatize(line[i])
                if line[i] == 'have':
                    line[i] = ''
                if i != j-1:
                    if line[i] != '' and line[i+1] != '':
                        line[i] = line[i]+' '
                else:
                    line[i] = line[i]+'\n'
                fw.write(line[i])
        f.close()
        fw.close()

def pretreat_ans():

    data_list= ['train']

    for data in data_list:
        
        fw = open('data/processed/' + data + '.ans', 'w')
        f = open('data/intermediate/' + data + '.a')
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '').split(' ')  
            j = len(line)
            for i in range(j): 
                for div in ['the','and','a','an','with']:
                    if line[i] == div:
                        line[i] = ''
                #if re.match(r'[0-9]+', line[i]):
                    #line[i] = 'num'
                #elif re.match(r'[a-z][0-9]+', line[i]):
                    #line[i] = 'pos'
                if line[i] != '':
                    line[i] = lemma.lemmatize(line[i])
                if line[i] == 'have':
                    line[i] = ''
                if i != j-1:
                    if line[i] != '' and line[i+1] != '':
                        line[i] = line[i]+' '
                else:
                    line[i] = line[i]+'\n'
                fw.write(line[i])
        f.close()
        fw.close()