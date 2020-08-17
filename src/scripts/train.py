import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from keras import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import *
from keras.callbacks import *

dim = 128
maxlen = 9

def create_model(ans_dic_ot, weight_matrixx_yn, weight_matrixx_ot, wrdidx_yn, wrdidx_ot, train_que_yn, train_que_ot, train_hot_yn, train_hot_ot):
    train_img_yn = np.load('data/vectorized/train_yn_im.npy')
    train_img_ot = np.load('data/vectorized/train_ot_im.npy')

    len_wrdidx_yn =len(wrdidx_yn) + 1
    len_wrdidx_ot =len(wrdidx_ot) + 1

    #image

    encoded_image = Input(shape=(1000,))
    dense_image = Dense(dim*2)(encoded_image)
    batch_image = BatchNormalization()(dense_image)
    repeat_image = RepeatVector(maxlen)(batch_image)

    encode_question_yn = Input(shape=(maxlen,))
    embed_question_yn = Embedding(len_wrdidx_yn,600,input_length=maxlen,weights=[weight_matrixx_yn],trainable=False)(encode_question_yn)
    batch_question_yn = Bidirectional(LSTM(dim, return_sequences=True,dropout=0.5))(embed_question_yn)

    dic = 1000
    encode_question_ot = Input(shape=(maxlen,))
    embed_question = Embedding(input_dim=dic, output_dim=dim, input_length=maxlen)(encode_question_ot)
    batch_question_ot = Bidirectional(LSTM(dim, return_sequences=True,dropout=0.5))(embed_question)

    #attention_yn
    merge_attention = add([repeat_image, batch_question_yn])
    act1_attention = Activation('tanh')(merge_attention)
    dense1_attention = TimeDistributed(Dense(1))(act1_attention)
    flat_attention = Flatten()(dense1_attention)
    act2_attention = Activation('softmax')(flat_attention)
    repeat_attention = RepeatVector(dim*2)(act2_attention)
    permute_attention_yn = Permute((2, 1))(repeat_attention)

    #attention_ot
    merge_attention = add([repeat_image, batch_question_ot])
    act1_attention = Activation('tanh')(merge_attention)
    dense1_attention = TimeDistributed(Dense(1))(act1_attention)
    flat_attention = Flatten()(dense1_attention)
    act2_attention = Activation('softmax')(flat_attention)
    repeat_attention = RepeatVector(dim*2)(act2_attention)
    permute_attention_ot = Permute((2, 1))(repeat_attention)

    #merge
    batch_model_yn = concatenate([batch_question_yn, permute_attention_yn])
    batch_model_yn = BatchNormalization()(batch_model_yn)

    batch_model_yn = Dense(3)(batch_model_yn)
    batch_model_yn = Flatten()(batch_model_yn)

    batch_model_ot = concatenate([batch_question_ot, permute_attention_ot])
    batch_model_ot = BatchNormalization()(batch_model_ot)

    batch_model_ot = Dense(maxlen)(batch_model_ot)
    batch_model_ot = Permute((2, 1))(batch_model_ot)

    #output model
    output_model_yn = Dense(3, activation='softmax')(batch_model_yn)
    vqa_model_yn = Model(inputs=[encoded_image,encode_question_yn], outputs=output_model_yn)
    vqa_model_yn.summary()

    output_model_ot = TimeDistributed(Dense(ans_dic_ot, activation='softmax'))(batch_model_ot)
    vqa_model_ot = Model(inputs=[encoded_image,encode_question_ot], outputs=output_model_ot)
    vqa_model_ot.summary()

    #compile
    adam = optimizers.Adam()

    vqa_model_yn.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=[metrics.binary_accuracy])
    vqa_model_ot.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=[metrics.categorical_accuracy])

    #save
    filepath_yn = 'models/MODEL_yn.hdf5'
    checkpoint_yn = ModelCheckpoint(filepath_yn, verbose=1, save_best_only=True, mode='min')
    callbacks_list_yn = [checkpoint_yn]

    filepath_ot = 'models/MODEL_ot.hdf5'
    checkpoint_ot = ModelCheckpoint(filepath_ot, verbose=1, save_best_only=True, mode='min')
    callbacks_list_ot = [checkpoint_ot]

    #train
    print('Training model yn...')
    history_yn = vqa_model_yn.fit([train_img_yn,train_que_yn],[train_hot_yn], epochs=300, batch_size=256, callbacks=callbacks_list_yn, verbose=1)

    model_json = vqa_model_yn.to_json()
    with open('models/MODEL_yn.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    vqa_model_yn.save_weights('models/MODEL_yn.h5')
    print("Saved model yn to disk")

    print('Training model ot...')
    history_ot = vqa_model_ot.fit([train_img_ot,train_que_ot],[train_hot_ot], epochs=300, batch_size=256, callbacks=callbacks_list_ot, verbose=1)

    model_json = vqa_model_ot.to_json()
    with open('models/MODEL_ot.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    vqa_model_ot.save_weights('models/MODEL_ot.h5')
    print("Saved model yn to disk")

    return(vqa_model_yn, vqa_model_ot)

def load_model():

    print('Loading models...')  

    #yn
    json_file = open('models/MODEL_yn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    vqa_model_yn = model_from_json(loaded_model_json)
    vqa_model_yn.load_weights('models/MODEL_yn.h5', by_name=True) 

    #ot
    json_file = open('models/MODEL_ot.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    vqa_model_ot = model_from_json(loaded_model_json)
    vqa_model_ot.load_weights('models/MODEL_ot.h5', by_name=True) 

    print("Loaded models from disk...")

    return(vqa_model_yn, vqa_model_ot)

    
