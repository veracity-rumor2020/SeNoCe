import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
#from tensorflow import set_random_seed
tf.random.set_seed(1)
#set_random_seed(1)#tf.random.set_seed(1)#
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import pickle
import pandas as pd
import keras
from keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
import sys
import pickle

import rumour_Loss as rL
import getXY_Loss as gxy

pout="repo_Fast/"
epoch = XXX
trainStatsFile=pout+"dfTrainStats.csv"
saveModel=pout+"vsrModel.h5"
predFile=pout+"predicted.pkl"

############################################changes aboves

def train_test():

	f=open("embeddingMatrix.pkl", "rb") #create Embedding matrix using googlenews word2vec embedding.
	embedding_Matrix=pickle.load(f)
	f.close()
	embedding_Matrix=embedding_Matrix[0]
	branchDim = XXX #maximum length of branch.
	src_len = 100 #maximum no of words in source tweet
	vocab_size = YYY # vocabulary size
	sD={"source":(src_len, ), "branch":(branchDim, ), "lstmOP":1, "lstmOP2":100, "dense_Size":100, "regularizer":0.0001}
	sD["vocab_size"]=vocab_size; sD["outdim"]=300; sD["input_length"]=branchDim
	sD["embedding_Matrix"]=embedding_Matrix
	
	##get data
	xtr, ytr, xts, yts = gxy.xyRet() #prepare training and testing data

	##get model
	model=rL.getVSModel(**sD)	

	##training & validation
	dfStats=pd.DataFrame(columns=["val_private_vsr_loss", "val_aux_vsr_loss", "val_private_vsr_acc", "val_aux_vsr_acc"])
	his=model.fit(xtr, ytr, epochs=epoch, batch_size=32, validation_split=0.2, verbose=2)
	statsL=[ his.history["val_private_vsr_loss"], his.history["val_aux_vsr_loss"], his.history["val_private_vsr_acc"], his.history["val_aux_vsr_acc"] ]
	#print (statsL)
	dfStats["val_private_vsr_loss"], dfStats["val_aux_vsr_loss"], dfStats["val_private_vsr_acc"], dfStats["val_aux_vsr_acc"]=statsL[0], statsL[1], statsL[2], statsL[3]
	dfStats.to_csv(trainStatsFile, index=False)

	#saving model
	model.save(saveModel)
	#model=load_model(saveModel, custom_objects={"tf":tf})

	pred=model.predict(xts)
	f=open(predFile, "wb")
	pickle.dump(pred, f)
	f.close()

	return


train_test()

	
