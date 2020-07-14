import os, sys
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def get_length(data):
    return(len(open('data/processed/'+data+'.name', 'r').readlines()))


def img_processing():

    model = InceptionResNetV2(weights='imagenet')  #The TF and Keras versions are related to this code.
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)

    data_list= ['train_yn','train_ot','test_yn','test_ot']

    for data in data_list:

        f = open('data/processed/' + data + '.name')
        im = []
        i = 0
        z = np.zeros((1,1000))
        z = z.tolist()
        while 1:
            lines = f.readlines()
            if not lines:
                break
            for line in lines:
                line = line.strip()
                if os.path.isfile('data/raw/images/' + line + '.jpg') == True:
                    img_path = 'data/raw/images/' + line + '.jpg'
                    img = load_img(img_path, target_size=(299, 299))
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x_o = preprocess_input(x)
                    y = model.predict(x_o)
                    y = y.tolist()
                    im.append(y)
                else:
                    im.append(z)
                    print('Picture NO.', i+1, ' could not be found.')
                i += 1
                
                num_records = get_length(data)

                num_arrow = int(i * 50 / num_records) + 1
                percent = i * 100.0 / (num_records)

                num_line = 50 - num_arrow
                process_bar = data +': [' + '>' * num_arrow + '-' * num_line + ']'\
                        + '%.2f' % percent + '%' + '\r'
                sys.stdout.write(process_bar)
                sys.stdout.flush()
        im = np.array(im)
        im = np.squeeze(im)
        np.save('data/vectorized/' + data + '_im.npy', im)
        print('\n')
        f.close()