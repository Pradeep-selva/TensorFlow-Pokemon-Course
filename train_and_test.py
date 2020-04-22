import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
import utilities as ut

dex_no = int(input("Enter the pokedex number to predict if its Legendary or not: "))-1

df = pd.read_csv('./pokemon/pokemon.csv')
names = df['Name']
df = df[['isLegendary', 'Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def','Speed', 'Color', 'Egg_Group_1', 'Height_m', 'Weight_kg', 'Body_Style']]

df['isLegendary'] = df['isLegendary'].astype(int)
df = ut.dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])

df_train, df_test = ut.train_test_splitter(df, 'Generation')
train_data, train_labels, test_data, test_labels = ut.label_delineator(df_train, df_test, 'isLegendary')
train_data, test_data = ut.data_normalizer(train_data, test_data)

length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=400)

print("----------------------------------------------------------------")
ut.predictor(model, test_data, test_labels, names,  dex_no)
print("----------------------------------------------------------------")
