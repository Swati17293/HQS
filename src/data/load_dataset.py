def process_files(): 

    data_list= ['train','test']

    for data in data_list:

        data_name = open('data/intermediate/' + data + '.name','w',encoding='utf-8')  
        data_que  = open('data/intermediate/' + data + '.q', 'w',encoding='utf-8')  
        data_ans  = open('data/intermediate/' + data + '.a', 'w',encoding='utf-8')  

        with open('data/raw/' + data + '.csv', encoding='utf-8') as f:
            for line in f:
                line_list = line.split('|')
                data_name.write(line_list[0]+'\n')
                data_que.write(line_list[1].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower()+'\n')
                data_ans.write(line_list[2].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower())
    
        data_name.close()
        data_que.close()
        data_ans.close()

    data_answer  = open('data/intermediate/test.answer', 'w',encoding='utf-8')
    with open('data/raw/test.csv', encoding='utf-8') as f:
        i = 1
        for line in f:
            line_list = line.split('|')
            line_ = str(i) + '  ' + line_list[0].strip() + '    ' + line_list[2]
            data_answer.write(line_)
            i += 1

    data_answer.close()
