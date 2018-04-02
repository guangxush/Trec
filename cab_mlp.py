from __future__ import print_function
from keras import losses
from keras import Sequential
from keras.layers import Dense,Dropout
from keas.metrics import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import mean_absolute_error
from util.dataset import load_data_v1, make_submission
import numpy as np
import sys

def mlp(sample_dim, nb_classes):
    model = Sequential()
    model.add(Dense(2048, kernel_intializer='glorot_uniform', activation='relu', input_dim=sample_dim))
    model.add(Dense(1024, kernal_intializer='glorot_uniform', activation='relu'))
    model.add(Dense(nb_classes, kernal_intializer='glorot_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train = True if sys.argv[1] == 'train' else False
    print('****** Start Finding the Passenger ******')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test = load_data_v1(datapath='data')
    print('Training MLP model ...')
    check_pointer = ModelCheckpoint(filepath='models/mlp.hdf5', verbose=1, save_best_only=True,
                                    save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    csv_logger = CSVLogger('logs/mlp')
    mlp_model = mlp(sample_dim=x_train.shape[1])
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])

    if train:
        print('Generate result ...')
        mlp_model.load_weights(filepath='models/mlp.hdf5')
        results = mlp_model.predict(x_test).reshape(-1,1)
        results_dev = mlp_model.predict(x_dev).reshape(-1,1)
        result_dev_new = []
        for item in results_dev:
            item_value = item[0]
            item_value = np.round(item_value)
            result_dev_value.append(item_value)
        result_dev_new = np.asarray(result_dev_new).reshape(-1,1)
        print('Dev MAE:', mean_absolute_error(y_dev, result_dev_new))
        make_submission(result_path='submissions', results=results, model_name='MLP')

    print('***** Found the Passenger ******')