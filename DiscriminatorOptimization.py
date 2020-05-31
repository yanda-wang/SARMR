import skorch
import pickle
import os
import torch.nn as nn
import numpy as np

from torch import optim
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import train_test_split

from Networks import DiscriminatorMLPPremiumForTuning
from Parameters import Params

params = Params()
HIDDEN_SIZE = params.HIDDEN_SIZE
NUM_CLASS = params.fitted_distribution_label_num
MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH
device = params.device
FITTED_DISTRIBUTION_FILE_NAME_TRAIN = params.fitted_distribution_file_name_train
FITTED_DISTRIBUTION_FILE_NAME_TEST = params.fitted_distribution_file_name_test
ENCODER_OUTPUT_FILE_NAME_TRAIN = params.ENCODER_OUTPUT_TRAIN_FILE_NAME
ENCODER_OUTPUT_FILE_NAME_TEST = params.ENCODER_OUTPUT_TEST_FILE_NAME

OPTIMIZATION_LOG_FILE = 'data/log/discriminatorTuning/optimization.log'


def split_data_distribution(distribution_data_file, encoder_output_data_file, output_file_path,
                            distribution_mode='single'):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    encoder_train_output_file = os.path.join(output_file_path, 'encoder_output_train')
    encoder_test_output_file = os.path.join(output_file_path, 'encoder_output_test')
    encoder_output = np.load(encoder_output_data_file)
    encoder_train, encoder_test = train_test_split(encoder_output, test_size=0.2)

    if distribution_mode == 'single':
        distribution_train_output_file = os.path.join(output_file_path, 'single_distribution_train')
        distribution_test_output_file = os.path.join(output_file_path, 'single_distribution_test')
        distribution_data = np.load(distribution_data_file)['real_data']  # np.array
        distribution_train, distribution_test = train_test_split(distribution_data, test_size=0.2)
    else:
        distribution_train_output_file = os.path.join(output_file_path, 'multi_distribution_train')
        distribution_test_output_file = os.path.join(output_file_path, 'multi_distribution_test')

        data = pickle.load(open(distribution_data_file, 'rb'),
                           encoding='latin1')  # dict, key=cluster label,value=np.array
        distribution_train, distribution_test = [], []
        for key, value in data.items():
            current_train, current_test = train_test_split(value, test_size=0.2)
            distribution_train.append(current_train)
            distribution_test.append(current_test)

        distribution_train = np.concatenate(distribution_train, axis=0)
        distribution_test = np.concatenate(distribution_test, axis=0)

    np.save(distribution_train_output_file, distribution_train)
    np.save(distribution_test_output_file, distribution_test)
    np.save(encoder_train_output_file, encoder_train)
    np.save(encoder_test_output_file, encoder_test)


def get_x_y(distribution_data_file, encoder_output_data_file):
    distribution_data = np.load(distribution_data_file)  # real data, label=1
    encoder_output = np.load(encoder_output_data_file)  # fake data,label=0

    label_distribution_data = np.ones((distribution_data.shape[0], 1))
    label_encoder_output = np.zeros((encoder_output.shape[0], 1))
    distribution_data = np.hstack((distribution_data, label_distribution_data))
    encoder_output = np.hstack((encoder_output, label_encoder_output))

    data = np.vstack((distribution_data, encoder_output))
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1]  # .reshape((-1, 1))

    print(Y.shape)

    X = X.astype('float32')
    Y = Y.astype('float32')

    return X, Y


def get_data(distribution_data_train_file, encoder_output_train_file, distribution_data_test_file,
             encoder_output_test_file):
    train_x, train_y = get_x_y(distribution_data_train_file, encoder_output_train_file)
    test_x, test_y = get_x_y(distribution_data_test_file, encoder_output_test_file)
    return train_x, train_y, test_x, test_y


def get_accuracy(y_predict, y_target):
    count = 0
    for y_p, y_t in zip(y_predict, y_target):
        if y_p == y_t:
            count += 1
    return float(count) / len(y_predict)


# search_space = [Real(low=0, high=1, name='dropout_rate'),
#                 Integer(low=1, high=10, name='n_hidden_layers'),
#                 Integer(low=32, high=150, name='dim_B'),
#                 Integer(low=32, high=150, name='dim_C'),
#                 Categorical(categories=['8', '16', '32', '64'], name='batch_size'),
#                 Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
#                 Real(low=1e-8, high=1e-2, prior='log-uniform', name='weight_decay')
#                 ]

search_space = [Real(low=0, high=1, name='dropout_rate'),
                Integer(low=1, high=10, name='n_hidden_layers'),
                Integer(low=32, high=150, name='dim_B'),
                Integer(low=32, high=150, name='dim_C'),
                Categorical(categories=['8', '16', '32', '64'], name='batch_size'),
                Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
                ]

current_count = 1
total_count = 25


@use_named_args(dimensions=search_space)
def fitness(dropout_rate, n_hidden_layers, dim_B, dim_C, batch_size, learning_rate):
    weight_decay = 0
    batch_size = int(batch_size)
    model = skorch.classifier.NeuralNetBinaryClassifier(module=DiscriminatorMLPPremiumForTuning,
                                                        module__hidden_size=HIDDEN_SIZE,
                                                        module__dropout_rate=dropout_rate,
                                                        module__n_hidden_layers=n_hidden_layers, module__dim_B=dim_B,
                                                        module__dim_C=dim_C, optimizer__lr=learning_rate,
                                                        optimizer__weight_decay=weight_decay,
                                                        criterion=nn.BCELoss,
                                                        optimizer=optim.Adam, max_epochs=MAX_EPOCH,
                                                        batch_size=batch_size,
                                                        callbacks=[
                                                            skorch.callbacks.ProgressBar(batches_per_epoch='auto'), ],
                                                        device=device, train_split=None)

    train_x, train_y, test_x, test_y = get_data(FITTED_DISTRIBUTION_FILE_NAME_TRAIN, ENCODER_OUTPUT_FILE_NAME_TRAIN,
                                                FITTED_DISTRIBUTION_FILE_NAME_TEST, ENCODER_OUTPUT_FILE_NAME_TEST)

    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)

    metric = get_accuracy(predict_y, test_y)
    print('*****************************')
    print('hyper pamameters:')
    print('dropout_rate:', dropout_rate)
    print('n_hidden_layers:', n_hidden_layers)
    print('dim_B:', dim_B)
    print('dim_C:', dim_C)
    print('batch_size:', batch_size)
    print('learning_rate:', learning_rate)
    print('weight_decay:', weight_decay)
    print('accuracy:', metric)

    log_file = open(OPTIMIZATION_LOG_FILE, 'a+')
    log_file.write('dropout_rate:' + str(dropout_rate) + '\n')
    log_file.write('n_hidden_layers:' + str(n_hidden_layers) + '\n')
    log_file.write('dim_B:' + str(dim_B) + '\n')
    log_file.write('dim_C:' + str(dim_C) + '\n')
    log_file.write('batch_size:' + str(batch_size) + '\n')
    log_file.write('learning_rate:' + str(learning_rate) + '\n')
    log_file.write('weight_decay:' + str(weight_decay) + '\n')
    log_file.write('accuracy:' + str(metric) + '\n')
    log_file.write('**************************************\n')
    log_file.close()

    return -metric


def optimize(n_calls):
    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True)
    print('*****************************')
    print('best result:')
    print('accuracy', -result.fun)
    print('optimal hyper-parameters:')

    log_file = open(OPTIMIZATION_LOG_FILE, 'a+')
    log_file.write('best result:\n')
    log_file.write(str(-result.fun) + '\n')
    log_file.write('optimal hyper-parameters\n')

    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)
        log_file.write(hyper)
        log_file.write(':')
        log_file.write(str(value))
        log_file.write('\n')

    log_file.close()


if __name__ == '__main__':
    optimize(10)
