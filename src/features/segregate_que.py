import pickle
from segregate_test import segg_test

def segg_train(que):

    que = que.strip()

    if que == 'yes' or que == 'no':
        return('yn')
    else:
        return('ot')

def train_que_seg():
    
    f_ans = open('data/processed/train.ans')
    lines_ans = f_ans.readlines()

    f_que = open('data/processed/train.que')
    lines_que = f_que.readlines()

    f_name = open('data/intermediate/train.name')
    lines_name = f_name.readlines()

    train_yn_ans = open("data/processed/train_yn.ans", "w")
    train_ot_ans = open("data/processed/train_ot.ans", "w")

    train_yn_que = open("data/processed/train_yn.que", "w")
    train_ot_que = open("data/processed/train_ot.que", "w")

    train_yn_name = open("data/processed/train_yn.name", "w")
    train_ot_name = open("data/processed/train_ot.name", "w")

    i = 0
    for line_ans in lines_ans:

        q_type = segg_train(line_ans)

        if q_type == 'yn':

            train_yn_ans.write(lines_ans[i])
            train_yn_que.write(lines_que[i])
            train_yn_name.write(lines_name[i])
            
        else:

            train_ot_ans.write(lines_ans[i])
            train_ot_que.write(lines_que[i])
            train_ot_name.write(lines_name[i])

        i +=1

    f_ans.close()
    f_que.close()
    f_name.close()
    train_yn_ans.close()
    train_ot_ans.close()
    train_yn_que.close()
    train_ot_que.close()
    train_yn_name.close()
    train_ot_name.close()
    

def test_que_seg():

    q_types = segg_test()

    f_que = open('data/processed/test.que')
    lines_que = f_que.readlines()

    f_name = open('data/intermediate/test.name')
    lines_name = f_name.readlines()

    f_a = open('data/intermediate/test.a')
    lines_a = f_a.readlines()

    test_yn_que = open("data/processed/test_yn.que", "w")
    test_ot_que = open("data/processed/test_ot.que", "w")

    test_yn_name = open("data/processed/test_yn.name", "w")
    test_ot_name = open("data/processed/test_ot.name", "w")

    test_yn_a = open("data/intermediate/test_yn.a", "w")
    test_ot_a = open("data/intermediate/test_ot.a", "w")

    i = 0
    test_yn, test_ot = [], []

    for q_type in q_types:

        if q_type == 'yn':
            
            test_yn_que.write(lines_que[i])
            test_yn_name.write(lines_name[i])
            test_yn.append(i)

            line = str(i+1) + '   ' + lines_name[i].strip() + '   ' + lines_a[i]
            test_yn_a.write(line)

        else:
            
            test_ot_que.write(lines_que[i])
            test_ot_name.write(lines_name[i])
            test_ot.append(i)

            line = str(i+1) + '   ' + lines_name[i].strip() + '   ' + lines_a[i]
            test_ot_a.write(line)

        i +=1

    with open('data/processed/test_yn.data', 'wb') as filehandle:
        pickle.dump(test_yn, filehandle)

    with open('data/processed/test_ot.data', 'wb') as filehandle:
        pickle.dump(test_ot, filehandle)

    f_que.close()
    f_name.close()
    f_a.close()
    test_yn_que.close()
    test_ot_que.close()
    test_yn_name.close()
    test_ot_name.close()
    test_yn_a.close()
    test_ot_a.close()
