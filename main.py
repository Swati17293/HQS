import os
import sys

path = os.getcwd()
sys.path.insert(1, path+'/src/data')
sys.path.insert(1, path+'/src/features')
sys.path.insert(1, path+'/src/scripts')

from load_dataset import process_files
from preprocess_text import pretreat_que, pretreat_ans
from segregate_que import train_que_seg, test_que_seg
from embed_image import img_processing
from load_features import *
from train import create_model, load_model
from generate_ans import generate_yn, generate_ot, generate_all

def main():

    print('\n\nGenerating data files...')
    if os.path.isfile('data/intermediate/train.num') == False:
        process_files()

    print('Preprocessing Text...')
    if os.path.isfile('data/processed/train.ans') == False:
        pretreat_que()
        pretreat_ans()

    print('Segregation...')
    if os.path.isfile('data/processed/train_yn.ans') == False:
        train_que_seg()
        test_que_seg()

    print('Image Embedding...')
    if os.path.isfile('data/vectorized/train_yn_im.npy') == False:
        img_processing()

    print('Building model...')

    train_que_yn, test_que_yn, weight_matrixx_yn, wrdidx_yn = get_que_features_yn()
    train_que_ot, test_que_ot, weight_matrixx_ot, wrdidx_ot = get_que_features_ot()

    train_hot_yn, train_hot_ot, ans_dic_ot, tokenizer_ = get_ans_features()

    if os.path.isfile('models/MODEL_yn.h5') == False:
        vqa_model_yn, vqa_model_ot = create_model(ans_dic_ot, weight_matrixx_yn, weight_matrixx_ot, wrdidx_yn, wrdidx_ot, train_que_yn, train_que_ot, train_hot_yn, train_hot_ot)
    else:
        vqa_model_yn, vqa_model_ot = load_model()

    print('Generating answers...')
    if os.path.isfile('results/test_yn.fn') == False:
        generate_yn(vqa_model_yn,test_que_yn)
        generate_ot(vqa_model_ot,test_que_ot,tokenizer_)
        generate_all()

    print('Finished...')
    print('\n\n')

if __name__ == "__main__":
    main()


