import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing


def dummy_creation(df,dummy_categories):

    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy], axis=1)
        df = df.drop(i,axis=1)
    return df


def train_test_splitter(df, column):

    df_train = df.loc[df[column] != 1]
    df_test = df.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return (df_train, df_test)


def label_delineator(df_train, df_test, label):

    train_data = df_train.drop(label, axis = 1).values
    train_labels = df_train[label].values

    test_data = df_test.drop(label, axis = 1).values
    test_labels = df_test[label].values

    return (train_data, train_labels, test_data, test_labels)


def data_normalizer(train_data, test_data):

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return (train_data, test_data)


def predictor(model, test_data, test_labels, names, index):
    prediction = model.predict(test_data)
    predicted, actual = str(np.argmax(prediction[index])), str(test_labels[index])
    if(int(predicted) == int(actual)):
        if int(predicted):
            predicted, actual = ["Legendary"]*2
        else:
            predicted, actual = ["Non-legendary"]*2
    else:
        if int(predicted):
            predicted = "Legendary"
            actual = "Non-Legendary"
        else:
            actual = "Legendary"
            predicted = "Non-Legendary"
    if np.argmax(prediction[index]) == test_labels[index]:
        print("Pokemon--{}, {} was correctly predicted to be a {}!".format(index+1, names[index], predicted))
    else:
        print("Pokemon--{}, {} was incorrectly predicted to be a {}, it was actually a {}".format(index+1, names[index], predicted, actual))
    
    return(prediction)



       