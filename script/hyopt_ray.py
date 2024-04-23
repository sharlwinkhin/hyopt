"""
Author: Shar Lwin Khin
Date: 23 April 2024

Reference: Ray Tune - https://docs.ray.io/en/latest/tune/index.html

Dataset: https://www.unb.ca/cic/datasets/

"""
import os, time, datetime, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from keras.layers import Dense, Dropout, RepeatVector
from keras.layers import Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam 

# Ray libraries
from ray import air, tune, ray
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB  # bohb
from ray.tune.search.optuna import OptunaSearch  # optuna
from ray.tune.search.nevergrad import NevergradSearch  # nevergrad
from ray.tune.search.dragonfly import DragonflySearch  # dragonfly
import nevergrad as ng
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler

# Ray Setup
os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"
if ray.is_initialized():
    ray.shutdown()
ray.init()

# fix random seed for reproducibility
tf.random.set_seed(7)

def save_plot_trials(trials, output_plot, output_file, title, params):
    temp = {}
    for i, trial in enumerate(trials):
        if i == 0:
            for key in trial.config:
                if not 'mean_acc' in key:
                    temp[key] = [trial.config[key]]
            temp['mean_acc'] = [trial.metric_analysis['mean_accuracy']['avg']]
        else:
            for key in trial.config:
                temp[key].append(trial.config[key])
            temp['mean_acc'].append(trial.metric_analysis['mean_accuracy']['avg'])
    df =  pd.DataFrame(temp)
    print(df.shape)
    df.to_csv(output_file)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(12, 15))
    fig.suptitle(title, fontsize=10, fontstyle='italic')

    xs = [i for i, trial in enumerate(trials)]
    ys = [trial.metric_analysis['mean_accuracy']['avg'] for trial in trials]
    ax1.plot(xs,ys)
    ax1.set_title(f'accuracy vs trial ', fontsize=16)
    ax1.set_xlabel('$trial$', fontsize=12)
   
    ys = [trial.config[params[0]] for trial in trials]
    ax2.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax2.set_title(f'{params[0]} vs trial ', fontsize=16)
    ax2.set_xlabel('$trial$', fontsize=12)
   
    ys = [trial.config[params[1]] for trial in trials]
    ax3.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax3.set_title(f'{params[1]} vs trial ', fontsize=16)
    ax3.set_xlabel('$trial$', fontsize=12)
   
    ys = [trial.config[params[2]] for trial in trials]
    ax4.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax4.set_title(f'{params[2]} vs trial ', fontsize=16)
    ax4.set_xlabel('$trial$', fontsize=12)

    ys = [trial.config[params[3]] for trial in trials]
    ax5.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax5.set_title(f'{params[3]} vs trial ', fontsize=16)
    ax5.set_xlabel('$trial$', fontsize=12)

    ys = [trial.config[params[4]] for trial in trials]
    ax6.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax6.set_title(f'{params[4]} vs trial ', fontsize=16)
    ax6.set_xlabel('$trial$', fontsize=12)

    ys = [trial.config[params[5]] for trial in trials]
    ax7.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax7.set_title(f'{params[5]} vs trial ', fontsize=16)
    ax7.set_xlabel('$trial$', fontsize=12)

    ys = [trial.config[params[6]] for trial in trials]
    ax8.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax8.set_title(f'{params[6]} vs trial ', fontsize=16)
    ax8.set_xlabel('$trial$', fontsize=12)

    ys = [trial.config[params[7]] for trial in trials]
    ax9.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax9.set_title(f'{params[7]} vs trial ', fontsize=16)
    ax9.set_xlabel('$trial$', fontsize=12)

    plt.savefig(output_plot)
    if showPlot:
        plt.show()
    plt.clf()
   
    
# Build the Model
def build_model(num_uniq_apis, n_columns, optimizer='adam', loss='binary_crossentropy', lstm_sz=128,  embedding_vector_length = 64, dropout = 0.2, model_name='lstm', n_hidden_layer=0, hidden_sz=128, output_sz=6):
    model = Sequential()
 
    if model_name == 'cnn-lstm':
        if isSequence:
            model.add(Embedding(num_uniq_apis, embedding_vector_length, input_length=n_columns))
            model.add(Dropout(dropout))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        else:
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'), input_shape=(n_columns,1))
    
        model.add(Dropout(dropout))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(lstm_sz))
        model.add(Dropout(dropout))
        for _ in range(n_hidden_layer): 
            model.add(Dense(hidden_sz, activation='relu'))
            model.add(Dropout(dropout))
        
    elif model_name == 'bilstm':
        if isSequence:
            model.add(Embedding(num_uniq_apis, embedding_vector_length, input_length=n_columns))
            model.add(Dropout(dropout))
            model.add(Bidirectional(LSTM(lstm_sz)))
        else:
            model.add(Bidirectional(LSTM(lstm_sz), input_shape = (n_columns,1)))
           
        model.add(Dropout(dropout))
        for _ in range(n_hidden_layer): 
            model.add(Dense(hidden_sz, activation='relu'))
            model.add(Dropout(dropout))
       
    elif model_name == 'lstm':
        if isSequence:
            model.add(Embedding(num_uniq_apis, embedding_vector_length, input_length=n_columns))
            model.add(Dropout(dropout))
            model.add(LSTM(lstm_sz)) 
        else:
            model.add(LSTM(lstm_sz, input_shape=(n_columns,1)))

        model.add(Dropout(dropout))
        for _ in range(n_hidden_layer): 
            model.add(Dense(hidden_sz, activation='relu'))
            model.add(Dropout(dropout))
       
    elif model_name == 'ann':
        if isSequence:
            model.add(Embedding(num_uniq_apis, embedding_vector_length, input_length=n_columns))
        else:
            model.add(Dense(hidden_sz, input_dim=n_columns, activation='relu'))
        model.add(Dropout(dropout))
        for _ in range(n_hidden_layer): 
            model.add(Dense(hidden_sz, activation='relu'))
            model.add(Dropout(dropout))

    # add output layer -- applicable to all classifiers
    if isMultiClass:
        print(f"output size: {output_sz}")
        model.add(Dense(output_sz, activation='softmax')) # non-sequence+multiclass
    else:
        model.add(Dense(1, activation='sigmoid')) # applies to both sequence+binary and non-sequence+binary 
        
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy']) #  metrics=['mean_squared_error']
   
    return model

def fit_network(doPlot, X, y, epochs, batch_size, lr, decay, momentum, loss, lstm_sz, embedding_vector_length, dropout, model_name, n_hidden_layer, hidden_sz):
    X = X.values # convert to numpy
    # the following shape the input data for different types
    if not isSequence:
        if not isMultiClass: # if it's multi-class y is already converted to numpy
            y = y.values
            if model_name == 'lstm' or model_name == 'bilstm':
                #print(y.shape)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                y = y.reshape((y.shape[0], 1))
        else:
            if model_name == 'lstm' or model_name == 'bilstm':
                print(f"printing y shape before: {y.shape}")
                X = X.reshape((X.shape[0], X.shape[1], 1))
                y = y.reshape((y.shape[0], y.shape[1], 1))
                print(f"printing y shape after: {y.shape}")
    else:
        if not isMultiClass:
            y = y.values
            if model_name == 'ann':
                y = np.asarray(y).astype('float32').reshape((-1,1))


    train_X, valid_X, train_Y, valid_Y = train_test_split(X, y, test_size=0.2, random_state=0)
   
    earlystop = EarlyStopping(
        monitor = 'val_accuracy', 
        min_delta = 0.0001, 
        patience = 20, 
        mode = 'max', 
        verbose = 1,
        restore_best_weights = True
    )
    callback = [earlystop]
   
    optimizer = Adam(learning_rate=lr,weight_decay=decay,ema_momentum=momentum)  
    
    model = build_model(num_uniq_apis=num_uniq_apis, n_columns=n_columns, optimizer=optimizer,loss=loss, lstm_sz=lstm_sz, embedding_vector_length=embedding_vector_length, dropout=dropout, model_name=model_name, n_hidden_layer=n_hidden_layer, hidden_sz=hidden_sz, output_sz=n_labels)

    # print(model.summary())
    # fit network
    history = model.fit(train_X, train_Y,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(valid_X, valid_Y),
                        verbose=2,
                        shuffle=True, callbacks=callback) # with callback   
    # model.save(f"{model_file}{counter[0]}") # if need to save the model
    
    # plot history
    if doPlot:
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='validation_loss')
        axis_font = {'size': '18'}
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        plt.legend(prop=axis_font)
        plt.savefig(loss_plot)
        plt.clf()
        plt.plot(history.history['val_accuracy'], label='validation_acc')
        axis_font = {'size': '18'}
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        plt.legend(prop=axis_font)
        plt.savefig(acc_plot)
        if showPlot:
            plt.show()
        plt.clf()

    # need to return these two values for optimizing search
    mean_train_loss = np.mean(history.history['loss'])
    mean_test_loss = np.mean(history.history['val_loss'])
    mean_test_acc = np.mean(history.history['val_accuracy'])

    return model, mean_test_loss, mean_test_acc, mean_train_loss

def objective(space):
    trial_counter[0] += 1
    print(f"Trial {trial_counter[0]}")
    model, test_loss, test_acc, train_loss = fit_network(doPlot=False, X=X_train, y=y_train, epochs = int(space['epochs']), batch_size = space['batch_size'], lr=space['lr'], decay=space['decay'], momentum=space['momentum'], loss=space['loss'], lstm_sz = space['lstm_sz'], embedding_vector_length = space['embedding_vector_length'], dropout = space['dropout'], model_name=space['model_name'], n_hidden_layer= int(space['n_hidden_layer']), hidden_sz=space['hidden_sz'])

    print(f"Mean Train Loss: {train_loss}. Mean Test Loss: {test_loss}. Mean Test Acc: {test_acc}")
    #write to a csv file
    with open(loss_file, 'a+') as f:
        current_time = datetime.datetime.now()
        writer = csv.writer(f)
        writer.writerow([space, test_loss, test_acc, current_time])

    # Send the current training result back to Tune
    tune.report(mean_accuracy=test_acc)
   

def evaluateBest(X_test,y_test,X_train, y_train, space):
    model, test_loss, test_acc, train_loss = fit_network(doPlot=True, X=X_train, y=y_train, epochs = int(space['epochs']), batch_size = space['batch_size'], lr=space['lr'], decay=space['decay'], momentum=space['momentum'], loss=space['loss'], lstm_sz = space['lstm_sz'], embedding_vector_length = space['embedding_vector_length'], dropout = space['dropout'], model_name=space['model_name'], n_hidden_layer= int(space['n_hidden_layer']), hidden_sz=space['hidden_sz'])
   
    # It can be used to reconstruct the model identically.
    # model = load_model(best)
    print(model.summary())

    if isSequence:
        if space['model_name'] == 'ann':
            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    y_pred = model.predict(X_test)
   
    if isMultiClass:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(y_test,y_pred)
        fmeasure = 0
        recall = 0
        precision = 0
        accuracy = balanced_accuracy_score(y_test, y_pred)
    else:
        if isSequence and space['model_name'] == 'ann':
            y_pred = np.argmax(y_pred, axis=1)
            
        preds3 = [ 1 if i>0.5 else 0 for i in y_pred  ] 
        fmeasure = f1_score(y_test, preds3)
        recall = recall_score(y_test, preds3)
        precision = precision_score(y_test, preds3)
        accuracy = balanced_accuracy_score(y_test, preds3)

    # print(fmeasure, recall, precision)
    return fmeasure, recall, precision, accuracy


def getAlgorithm(algo_name):
    # print(f"Running {algo_name} Algorithm")
    scheduler = ASHAScheduler()  # default scheduler
    if algo_name == "hyperopt":
        return HyperOptSearch(), scheduler

    if algo_name== "optuna":
        return OptunaSearch(), scheduler
    
    if algo_name == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=False,
        )
        return TuneBOHB(), scheduler
    
    if algo_name == "nevergrad":
        return NevergradSearch(optimizer=ng.optimizers.OnePlusOne), scheduler

    if algo_name == "dragonfly":
        return (
            DragonflySearch(
                optimizer="random",
                domain="euclidean",
            ),
            scheduler,
        )
    
    if algo_name == "bayesopt":
        return BayesOptSearch(), scheduler
   
if __name__ == '__main__':
    start_time = time.time()
    # set some flags
    isMultiClass = False
    isSequence = False
    pdfMalware = False
    
    algo_choice = 0
    data_choice = 0
    ################################
    datasets = [ '../dataset/CICDataset/pdf/csv/PDFMalware2022_processed_shuffle', 
                '../dataset/CICDataset/MalDroid-2020/csv/feature_vectors_syscallsbinders_frequency_5_Cat_shuffle', 
                '../dataset/msoft/perm_seq_shuffle',  
                '../dataset/msoft/pack_freq_shuffle', 
                '../dataset/dos/dos_shuffle', 
                '../dataset/IoTCryptojacking/crypto1_shuffle',
                '../dataset/IoTCryptojacking/crypto2_shuffle'
                ]
    names = ['pdfmalware', 'maldroid', 'permseq', 'packfreq', 'dos', 'crypto1', 'crypto2']
    
    algorithms = ["rangrid", "dragonfly", "bohb", "bayesopt", "hyperopt", "optuna","nevergrad"]
    
    input_file = datasets[data_choice]
    name = names[data_choice]

    debug = False
    showPlot = False
    n_trials = 500
    ratio = 0.8
    cv = 3  # cross validation
    local_dir = "ray_results"
    result = 'result.csv'
    label_name = 'Class'
    trial_counter = {}
    
    # search space
    if isSequence:
        model_names = ['lstm', 'bilstm', 'cnn-lstm']
    else:
        model_names = ['ann']
    epoch_list = np.arange(1, 100, 1, dtype=int)
    batch_sizes =  [128, 256, 512, 1024]
    optimizers = ['adam'] 
    if isMultiClass:
        loss_fns = ['categorical_crossentropy']
    else:
        loss_fns = ['binary_crossentropy']
    lstm_sizes = [64]
    embedding_vector_lengths = [64]
    if isSequence:
        lstm_sizes = [32, 64, 128, 256, 512]
        embedding_vector_lengths = [32, 64, 128, 256, 512]
    hidden_sizes = [32, 64, 128, 256, 512]

    # # define search space
    space = {
        'epochs': tune.choice(epoch_list),
        'n_hidden_layer': tune.quniform(1,5,1),  # [0, 1,2,3, ...]
        'hidden_sz': tune.choice(hidden_sizes),  # number of neurons in hidden layer
        'dropout': tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "decay": tune.loguniform(1e-4, 1e-1), # 'weight_decay'
        'batch_size': tune.choice(batch_sizes), 
        'lstm_sz': tune.choice(lstm_sizes), 
        'model_name': tune.choice(model_names),  
        'embedding_vector_length': tune.choice(embedding_vector_lengths),
        'loss' : tune.choice(loss_fns)
    }
    cols = ['epochs', 'n_hidden_layer', 'hidden_sz', 'dropout',   'lr', 'momentum', 'decay', 'batch_size', 'model_name', 'lstm_sz', 'embedding_vector_length', 'loss', 'mean_acc']
       
    for algo_name in algorithms:
        print(f"Running {algo_name} Algorithm. #Trial: {n_trials}")

        for i in range(cv):
            fmeasures = []; recalls = []; precisions = []; accuracies = []; val_accuracies = []; bests = []
            # specifies files ########################################
            model_file = f'ray_results/{name}_{algo_name}_cv{i}'  # to save the model
            best_file = f'ray_results/best/{name}_{algo_name}_cv{i}_best.csv'
            loss_file = f'{name}_{algo_name}_cv{i}_loss.csv'
            trial_file = f'ray_results/trial/{name}_{algo_name}_trial'
            loss_plot = f'ray_results/loss/{name}_{algo_name}_cv{i}_loss.pdf'
            acc_plot = f'ray_results/loss/{name}_{algo_name}_cv{i}_acc.pdf'
            trial_plot = f'ray_results/trial/{name}_{algo_name}_trial'
            ##########################################################
            trial_counter[0] = 0
            data = pd.read_csv(f"{input_file}{i+1}.csv", header=0, index_col=0)
            # n_malware = len(data[data[label_name]==1])
            n_samples = len(data)
            #print(data.shape, n_malware)
            data = data.dropna(axis=1, how='any') 
            X = data.drop(label_name, axis=1)

            if pdfMalware:
                X = pd.get_dummies(X, columns=['text', 'header'], dtype='int')
           
            num_uniq_apis = X.max().max() + 1 # get max values of all columns -- for embedding purpose
            n_columns = len(X.columns)
            y = data[label_name]
            n_labels = len(np.unique(y))
            print(n_columns, n_labels)

            split_index = round(n_samples * ratio)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

            if isMultiClass:
                y_train = to_categorical(y_train) 
                y_test = to_categorical(y_test) 
                n_labels = y_train.shape[1]
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    
            start_time = time.time()
            if algo_name != "rangrid":
                algo, scheduler = getAlgorithm(algo_name)
                analysis = tune.run(
                    objective,
                    name=algo_name,
                    local_dir=local_dir,
                    num_samples=n_trials,
                    config=space,
                    scheduler=scheduler,
                    search_alg=algo,
                    metric="mean_accuracy",
                    mode="max"
                )
            else:
                analysis = tune.run(
                    objective,
                    name=algo_name,
                    local_dir=local_dir,
                    num_samples=n_trials,
                    config=space,
                    metric="mean_accuracy",
                    mode="max"
                )

            end_time = time.time()
            duration = end_time - start_time

            output_plot = f'{trial_plot}_cv{i+1}.pdf'
            output_file = f'{trial_file}_cv{i+1}.csv'

            save_plot_trials(analysis.trials, output_plot, output_file, algo_name, cols)
            # print(best)
            # pickle.dump(trials, open(save_trials, "wb"))
            best = analysis.best_trial.config
            best['mean_acc'] = analysis.best_trial.metric_analysis['mean_accuracy']['avg']
            pd.DataFrame.from_dict(best, orient='index', columns=['value']).to_csv(best_file)
            fmeasure, recall, precision, accuracy = evaluateBest(X_test,y_test,X_train, y_train, best)
            print(f"F1: {fmeasure}. Recall: {recall}. Precision: {precision}. Accuracy: {accuracy}")
            fmeasures.append(fmeasure)
            recalls.append(recall)
            precisions.append(precision)
            accuracies.append(accuracy)
            val_accuracies.append(best['mean_acc'])
            bests.append(best)

        ############## compute means #################
        idx = np.argmax(val_accuracies)  # use val_accuracies as the metric
        mean_acc = np.mean(accuracies)
        mean_fmeasure = np.mean(fmeasures)
        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        # counter[0] = idx
        best = bests[idx]
        val_acc = val_accuracies[idx]
        print(best)

        with open(result, "a+") as f:
            timeTaken = round((time.time() - start_time)/60,2)  # in minutes
            current_time = datetime.datetime.now()
            f.write(f"\n{input_file}, {algo_name}, {n_samples}, {n_columns}, {mean_recall}, {mean_precision}, {mean_fmeasure}, {mean_acc}, {val_acc}, {n_trials}, {timeTaken}, {current_time}, {best}")


