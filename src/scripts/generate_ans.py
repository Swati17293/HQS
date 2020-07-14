import numpy as np
import pickle

maxlen =9

def generate_yn(vqa_model,test_que_yn):

    test_img_yn = np.load('data/vectorized/test_yn_im.npy')
    num = len(open('data/processed/test_yn.name', 'r').readlines())

    ans = vqa_model.predict([test_img_yn, test_que_yn])
    

    fp = open('results/test_yn_tmp.fn', 'w')

    for h in range(num):
        i = h
    
        if np.argmax(ans[i]) == 2:
            fp.write('yes\n')
        else:
            fp.write('no\n')
            
    fp.close()

def generate_ot(vqa_model,test_que_ot,tokenizer_a):

    test_img_ot = np.load('data/vectorized/test_ot_im.npy')
    num = len(open('data/processed/test_ot.name', 'r').readlines())

    dic_a = tokenizer_a.word_index
    ind_a ={value:key for key, value in dic_a.items()}

    ans = vqa_model.predict([test_img_ot, test_que_ot])

    fp = open('results/test_ot_tmp.fn', 'w')

    for h in range(num):
        i = h
        if np.argmax(ans[i][0],axis=0) == 0:
            fp.write('abnormality\n')
        else:
            try:
                for j in range(maxlen):
                    an = np.argmax(ans[i][j],axis=0) 
                    if j != maxlen-1:
                        anext = np.argmax(ans[i][j+1],axis=0)
                        if an != 0 and anext != 0:  
                            if an != anext:
                                fp.write(ind_a[an] + ' ')
                        elif an != 0 and anext == 0:  
                            fp.write(ind_a[an])
                        elif an == 0 and anext != 0:  
                            fp.write(' ')
                    else:
                        if an != 0:
                            fp.write(ind_a[an] + '\n')
                        else:
                            fp.write('\n')
            except:
                fp.write('abnormality\n')
    fp.close()

def generate_all():

    with open('data/processed/test_yn.data', 'rb') as filehandle:
        test_yn = pickle.load(filehandle)
    
    with open('data/processed/test_ot.data', 'rb') as filehandle:
        test_ot = pickle.load(filehandle)

    answer_test = {}

    f = open('results/test_yn_tmp.fn')
    f1 = f.readlines()
    for i in range(len(test_yn)):
        answer_test[test_yn[i]] = f1[i]
    f.close()
        
    f = open('results/test_ot_tmp.fn')
    f1 = f.readlines()
    for i in range(len(test_ot)):
        answer_test[test_ot[i]] = f1[i]
    f.close()

    #yn
    f = open('results/test_yn.fn', 'w')
    f1 = open('data/intermediate/test_yn.a')
    f2 = open('results/test_yn_tmp.fn')

    line2 = f2.readlines()

    i = 0
    for lines in f1:
        line1 = lines.split()

        line = line1[0] + '	' + line1[1] + '	' +  line2[i]
        i += 1
        f.write(line)
            
    f.close()
    f1.close()
    f2.close()

    #ot
    f = open('results/test_ot.fn', 'w')
    f1 = open('data/intermediate/test_ot.a')
    f2 = open('results/test_ot_tmp.fn')

    line2 = f2.readlines()

    i = 0
    for lines in f1:
        line1 = lines.split()
        line = line1[0] + '	' + line1[1] + '	' +  line2[i]
        i += 1
        f.write(line)
            
    f.close()
    f1.close()
    f2.close()

    f = open('results/test.fn', 'w')
    f1 = open('data/intermediate/test.answer')
    i = 0
    for lines in f1:
        line1 = lines.split()
        line = line1[0] + '	' + line1[1] + '	' +  answer_test[i]
        i += 1
        f.write(line)
            
    f.close()
    f1.close()

