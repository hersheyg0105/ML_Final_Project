"""
    MobileNet-v1 model written in TensorFlow Keras
"""
from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, \
    DepthwiseConv2D
from tensorflow.keras.models import Sequential
import numpy as np

def MobileNetv1():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool'))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def remove_channel(model):
    '''
    Input: model
           description: the pruned model
    Ouput: new_model
           description: the new model generating by removing all-zero channels
    '''
    def create_model(model):
        new_model = Sequential()
        score_list = np.sum(np.abs(model.layers[0].get_weights()[0]), axis=(0,1,2))
        next_layer_score_list = np.sum(np.abs(model.layers[0+3].get_weights()[0]), axis=(0,1,3))
        score_list = score_list * next_layer_score_list
        out_planes_num = int(np.count_nonzero(score_list))

        new_model.add(Conv2D(out_planes_num, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))

        new_model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))

        for i in range(6,75,6):
            if i == 78:
                pass
            else:
                if isinstance(model.layers[i],Conv2D):
                    old_strides = model.layers[i+3].strides
                    score_list = np.sum(np.abs(model.layers[i].get_weights()[0]), axis=(0,1,2))
                    next_layer_score_list = np.sum(np.abs(model.layers[i+3].get_weights()[0]), axis=(0,1,3))
                    score_list = score_list * next_layer_score_list
                    out_planes_num = int(np.count_nonzero(score_list))
                    out_planes_idx = np.squeeze( np.nonzero(score_list))
                            
                    new_model.add(Conv2D(out_planes_num, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
                    new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
                    new_model.add(Activation('relu'))

                    new_model.add(DepthwiseConv2D((3, 3), strides=old_strides, padding='same', use_bias=False))
                    new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
                    new_model.add(Activation('relu'))

        score_list = np.sum(np.abs(model.layers[78].kernel), axis=(0,1,2))
        out_planes_num = int(np.count_nonzero(score_list))

        new_model.add(Conv2D(out_planes_num, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))
        new_model.add(AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool'))
        new_model.add(Flatten())
        new_model.add(Dense(10))
        new_model.add(Activation('softmax'))
        return new_model

    new_model = create_model(model)

    def update_model(new_model, model):
        score_list = np.sum(np.abs(model.layers[0].get_weights()[0]), axis=(0,1,2))
        next_layer_score_list = np.sum(np.abs(model.layers[0+3].get_weights()[0]), axis=(0,1,3))
        score_list = score_list * next_layer_score_list
        out_planes_idx = np.squeeze( np.nonzero(score_list))
        old_wgt=model.layers[0].get_weights()[0]

        new_model.layers[0].set_weights([old_wgt[:,:,:,out_planes_idx]])
        old_wgt=model.layers[3].get_weights()[0]
        new_model.layers[3].set_weights([old_wgt[:,:,out_planes_idx,:]])
        input_planes_index = out_planes_idx
        for i in range(6,75,6):
            if i == 78:
                pass
            else:
                if isinstance(model.layers[i],Conv2D):
                    old_strides = model.layers[i+3].strides
                    score_list = np.sum(np.abs(model.layers[i].get_weights()[0]), axis=(0,1,2))
                    next_layer_score_list = np.sum(np.abs(model.layers[i+3].get_weights()[0]), axis=(0,1,3))
                    score_list = score_list * next_layer_score_list
                    out_planes_idx = np.squeeze( np.nonzero(score_list))

                    old_wgt=model.layers[i].get_weights()[0]
                    new_model_weigths=new_model.layers[i].get_weights()[0]
                    for idx,idx_out in enumerate(out_planes_idx):
                        new_model_weigths[:,:,:, idx] = old_wgt[:,:,input_planes_index,idx_out]
                    new_model.layers[i].set_weights([new_model_weigths])
                    old_wgt=model.layers[i+3].get_weights()[0]
                    new_model.layers[i+3].set_weights([old_wgt[:,:,out_planes_idx,:]])
                    input_planes_index = out_planes_idx
        score_list = np.sum(np.abs(model.layers[78].get_weights()[0]), axis=(0,1,2))
        out_planes_idx = np.squeeze( np.nonzero(score_list))


        new_model_weigths = new_model.layers[78].get_weights()[0]
        old_wgt=model.layers[78].get_weights()[0]
        for idx, idx_out in enumerate(out_planes_idx):
            new_model_weigths[:, :, :, idx] = old_wgt[:, :, input_planes_index, idx_out]
        new_model.layers[78].set_weights([new_model_weigths])

        old_wgt=model.layers[83].get_weights()
        new_model.layers[83].set_weights([old_wgt[0][out_planes_idx,:], old_wgt[1]])

        return new_model
    new_model=update_model(new_model,model)
    return new_model
