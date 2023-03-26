import os
import pandas as pd
import tensorflow as tf

import utils.helpers

# parse required arguments 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_config", help = "model configuration", type = str, default = 'CNN')
parser.add_argument("--epochs", help = "number of epochs", type = int, default = 10)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 128)
parser.add_argument("--augment", help = "data augmentation", type=bool, default = False)
args = parser.parse_args()

model_config = args.model_config
epochs = args.epochs
batch_size = args.batch_size
augment = args.augment

# read in the data
train_df = pd.read_csv('./Training_set.csv')
train_path = './train/'

# split the data into training and validation sets
x_train, x_test, y_train, y_test = utils.helpers.split_data(train_df, train_path, random_state=29)

# train the model and obtain a history
model, hist1, hist2 = utils.helpers.train_model(model_config, 
                                                x_train, 
                                                y_train, 
                                                x_test, 
                                                y_test, 
                                                augment=augment,
                                                batch_size=batch_size, 
                                                epochs=epochs)

# make directories where the results can be saved if they do not
# already exist
try:
    os.mkdir('./visualizations')
except:
    pass
try:
    os.mkdir('./models')
except:
    pass

# save the results 
model.save('./models/'+model_config+'.h5')

utils.helpers.plot_history(hist1, model_config)
