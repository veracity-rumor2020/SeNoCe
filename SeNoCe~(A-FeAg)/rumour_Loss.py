# Multiple Inputs
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
#tf.random.set_seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import keras
from keras import backend as k
from keras.layers.core import Lambda
from keras.layers import ConvLSTM2D, Conv2D, Dot, Concatenate, Flatten, Input, TimeDistributed, Embedding, BatchNormalization
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import load_model
from keras.layers.merge import concatenate
import math
from keras.optimizers import Adam
opt=Adam(lr=0.000001)

'''keras.regularizers.l1(0.01)
keras.regularizers.l2(0.01)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)'''
from keras.regularizers import l1, l2, l1_l2
#l1N=0.01; l2N=0.01; l1_l2N=0.01

np.random.seed(0)

def checkModel(model):
	names = [weight.name for layer in model2.layers for weight in layer.weights]#; print names
	weights = model2.get_weights()
	#d={str(names[i]).split("/")[0]:k.variable(weights[i]) for i in range(len(names)) if i%2==0}#; print d.keys()
	d={str(names[i]).split("/")[0]:weights[i] for i in range(len(names)) if i%2==0}#; print d.keys()
	#print "**********\n",k.eval(d["s_output"]), "\n"; input("enter a number")
	return d

def madeZ(x):
	s=x[int(len(x)/2):]
	x=x[:int(len(x)/2)]
	x=[x[i]*s[i] for i in range(len(x))]
	x=tf.add_n(x)
	return x

def scaledAttZ(x):
	s=x[:int(len(x)/2)]
	x=x[int(len(x)/2):]
	x=tf.nn.softmax(x)
	x=[x[i, ::] for i in range(x.shape[0])]
	x=[x[i]*s[i] for i in range(len(x))]
	x=tf.add_n(x)
	return x

def vecT(x):
	l=x.get_shape().as_list()
	x=tf.reshape(x, [-1, l[1]*l[2]])
	return x

def preLSTM(x):
	l=x.get_shape().as_list()
	x=tf.reshape(x, [-1, l[1], l[2]*l[3]])
	return x

def stanceModel(**kwargs):

	source=kwargs["source"]
	branch=kwargs["branch"]
	dense_Size=kwargs["dense_Size"]
	lstmOP2=kwargs["lstmOP2"]
	vocab_size=kwargs["vocab_size"]
	outdim=kwargs["outdim"]
	input_length=kwargs["input_length"]
	embedding_Matrix=kwargs["embedding_Matrix"]
	l2N=kwargs["regularizer"]

	visibleSrc=Input(shape=source)
	src=Embedding(input_dim=vocab_size, output_dim=outdim, weights=[embedding_Matrix], input_length=source[0], trainable=False)(visibleSrc)

	src=Lambda(vecT)(src)
	src=Dense(lstmOP2, activation="relu", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(src)

	visible=Input(shape=branch)
	s=Embedding(input_dim=vocab_size, output_dim=outdim, weights=[embedding_Matrix], input_length=branch[0], trainable=False)(visible)#; print ("embedding ", s.shape)
	s=LSTM(lstmOP2, activation="relu", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), recurrent_regularizer=l2(l2N), dropout=0.5, recurrent_dropout=0.5)(s)#; print ("lstm s **********", s.shape); input("enter a num")
	s=BatchNormalization()(s)

	#output to stances
	stances=4#; print ("shape s src ", s.shape, src.shape)
	sc=Concatenate(axis=1)([s, src])#; print sc.shape
	s_1=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(sc)
	s_2=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(sc)
	s_3=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(sc)
	s_4=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(sc)

	##orthogonal constraints
	ortho1=Dot(1, normalize=True)([s_1, s_2])#; print ortho1.shape
	ortho1=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho1)

	ortho2=Dot(1, normalize=True)([s_1, s_3])
	ortho2=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho2)

	ortho3=Dot(1, normalize=True)([s_1, s_4])
	ortho3=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho3)

	#ortho4=Dot(1, normalize=True)([s_1, s_5])
	#ortho4=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho4)

	ortho5=Dot(1, normalize=True)([s_2, s_3])
	ortho5=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho5)

	ortho6=Dot(1, normalize=True)([s_2, s_4])
	ortho6=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho6)

	#ortho7=Dot(1, normalize=True)([s_2, s_5])
	#ortho7=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho7)

	ortho8=Dot(1, normalize=True)([s_3, s_4])
	ortho8=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho8)

	#ortho9=Dot(1, normalize=True)([s_3, s_5])
	#ortho9=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho9)

	#ortho10=Dot(1, normalize=True)([s_4, s_5])
	#ortho10=Dense(1, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(ortho10)

	#stance_output=Dense(branch[0], activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s)#; print "stance ", stance_output.shape; input("enter a number")
	stance_output=Concatenate()([s_1, s_2, s_3, s_4])
	stance_preout=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(stance_output)
	stance_out=Dense(3, activation="softmax", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), name="private_vsr")(stance_output)

	stanceOPL=[ortho1, ortho2, ortho3, ortho5, ortho6, ortho8]
	#stanceOPL=[ortho1, ortho2, ortho3, ortho4, ortho5, ortho6, ortho7, ortho8, ortho9, ortho10]

	model=Model([visibleSrc, visible], [stance_output, ortho1, ortho2, ortho3, ortho5, ortho6, ortho8])

	#print model summary
	##print(model.summary())
	#plot graph
	plot_model(model, to_file='repo/stanceBranch.png')

	return visible, visibleSrc, stance_preout, stance_out, model, s_1, s_2, s_3, s_4, stanceOPL


def veracityModel(**kwargs):

	branch=kwargs["branch"]
	
	lstmOP=kwargs["lstmOP"]
	dense_Size=kwargs["dense_Size"]
	lstmOP2=kwargs["lstmOP2"]
	
	vocab_size=kwargs["vocab_size"]
	outdim=kwargs["outdim"]
	input_length=kwargs["input_length"]
	embedding_Matrix=kwargs["embedding_Matrix"]
	l2N=kwargs["regularizer"]

	(s_1, s_2, s_3, s_4)=kwargs["attentionTuple"]	

	visible=Input(shape=branch)
	s=Embedding(vocab_size, outdim,  input_length=branch[0], weights=[embedding_Matrix], trainable=False)(visible)
	s_en=LSTM(lstmOP2, activation="relu", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), dropout=0.5, recurrent_dropout=0.5)(s)
	s_en=BatchNormalization()(s_en)
	s_en=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s_en)
	#s_en=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s_en)
	
	##attention starts
	a1=Concatenate()([s_en, s_1])
	a1=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(a1)
	#a1=Dense(1, activation="sigmoid")(a1)

	a2=Concatenate()([s_en, s_2])
	a2=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(a2)
	#a2=Dense(1, activation="sigmoid")(a2)

	a3=Concatenate()([s_en, s_3])
	a3=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(a3)
	#a3=Dense(1, activation="sigmoid")(a3)

	a4=Concatenate()([s_en, s_4])
	a4=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(a4)
	#a4=Dense(1, activation="sigmoid")(a4)


	#a5=Concatenate()([s_en, s_5])
	#a5=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(a5)
	#a5=Dense(1, activation="sigmoid")(a5)	

	z=Lambda(scaledAttZ)([s_1, s_2, s_3, s_4]+[a1, a2, a3, a4])

	##both dense will have equal number of output dim equaling veracity label
	z=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(z)
	#z=Dense(3, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(z)
	z_out=Dense(3, activation="softmax", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), name="aux_vsr")(z)

	z_output=[z]

	return visible, z, z_out, z_output

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

def veracitySharedModel(**kwargs):

	branch=kwargs["branch"]
	
	lstmOP=kwargs["lstmOP"]
	dense_Size=kwargs["dense_Size"]
	lstmOP2=kwargs["lstmOP2"]
	
	vocab_size=kwargs["vocab_size"]
	outdim=kwargs["outdim"]
	input_length=kwargs["input_length"]
	embedding_Matrix=kwargs["embedding_Matrix"]
	l2N=kwargs["regularizer"]

	vsr_preOut=kwargs["vsr_preOut"]
	stance_preOut=kwargs["stance_preOut"]

	visible=Input(shape=branch)
	s=Embedding(vocab_size, outdim,  input_length=branch[0], weights=[embedding_Matrix], trainable=False)(visible)
	s_en=LSTM(lstmOP2, activation="relu", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), dropout=0.5, recurrent_dropout=0.5)(s)
	s_en=BatchNormalization()(s_en)
	s_en=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s_en)
	#s_en=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s_en)
	##both dense will have equal number of output dim equaling veracity label
	z=Dense(dense_Size, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(s_en)
	#z=Dense(3, activation="sigmoid", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N))(z)
	z=Lambda(grad_reverse)(z)
	z=Concatenate()([z, vsr_preOut, stance_preOut])
	z_out=Dense(3, activation="softmax", kernel_regularizer=l2(l2N), bias_regularizer=l2(l2N), name="shared_vsr")(z)
	z_output=[z]
	return visible, z, z_out, z_output

##{"source":(100, ), "branch":(48, ), "lstmOP":1, "lstmOP2":100, "dense_Size":100, "regularizer":0.01, "vocab_size":1000, "outdim":20, "input_length":48}
## + "attentionTuple":(s_1, s_2, ..s_4)

def getModels():

	sD={"source":(100, ), "branch":(48, ), "lstmOP":1, "lstmOP2":100, "dense_Size":100, "regularizer":0.01}
	sD["vocab_size"]=1000; sD["outdim"]=20; sD["input_length"]=48
	sD["embedding_Matrix"]=np.random.uniform(low=0, high=100, size=(1000, 20)).astype(int)
	visibleStance, visibleSrc, stance_preOut, stance_out, model, s_1, s_2, s_3, s_4, stanceOPL=stanceModel(**sD)
	
	
	vD={"branch":(48, ), "lstmOP":1, "lstmOP2":100, "dense_Size":100, "attentionTuple":(s_1, s_2, s_3, s_4), "regularizer":0.01}
	vD["vocab_size"]=1000; vD["outdim"]=20; vD["input_length"]=48
	vD["embedding_Matrix"]=np.random.uniform(low=0, high=10000, size=(1000, 20)).astype(int)
	visibleVeracity, z, z_out, veracity_output=veracityModel(**vD)

	vD["vsr_preOut"]=z; vD["stance_preOut"]=stance_preOut
	visibleVeracity_Shared, z_Shared, z_out_Shared, veracity_output_Shared=veracitySharedModel(**vD)	
	
	model=Model([visibleVeracity, visibleStance, visibleSrc, visibleVeracity_Shared], [z_out, stance_out, z_out_Shared]+stanceOPL)
	#print model summary
	print(model.summary())
	#plot graph
	plot_model(model, to_file='repo/branch.png')
	
	return model#, model_R

#getModels()


#*************************

def getVSModel(**sD):
	visibleStance, visibleSrc, stance_preOut, stance_out, model, s_1, s_2, s_3, s_4, stanceOPL=stanceModel(**sD)	
	
	sD["attentionTuple"]=(s_1, s_2, s_3, s_4)
	visibleVeracity, z, z_out, veracity_output=veracityModel(**sD)

	sD["vsr_preOut"]=z; sD["stance_preOut"]=stance_preOut
	visibleVeracity_Shared, z_Shared, z_out_Shared, veracity_output_Shared=veracitySharedModel(**sD)	
	
	model=Model([visibleSrc, visibleStance], [stance_out]+stanceOPL)
	model.compile(optimizer=opt, loss=['categorical_crossentropy', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error'], loss_weights=[0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], metrics=['accuracy'])
	#print model summary
	print(model.summary())
	#plot graph
	plot_model(model, to_file='repo/vsModel.png')
	
	return model


#####for stances check
def stanceCos(**kwargs):
	xL=kwargs["xL"]
	modelName=kwargs["modelName"]
	saveDf=kwargs["saveDf"]
	sReal=kwargs["sReal"]

	#model_vs, model_r=getModels()
	#model_vs.save("repo/tempo/stanceCheck.h5")
	model=load_model(modelName, custom_objects={'tf': tf})
	print ("model successfully loaded")#; input("enter")
	
	n=["vsrAttLSTM", "vsrAttOr"]
	inp=model.input
	#print ("type ", type(inp), type(inp[0]), len(inp), [x.shape for x in inp])
	outputs=[layer.output for layer in model.layers]# if n[0] in layer.output.name or n[1] in layer.output.name]
	outputs=[outputs[6], outputs[11], outputs[12], outputs[13], outputs[14]]
	#print ("outputs ", type(outputs), len(outputs), [x.name for x in outputs])
	functors = k.function([inp, k.learning_phase()], outputs)	

	model2=Model(inputs=inp, outputs=outputs)
	pred=model2.predict(xL)
	df=pd.DataFrame(columns=["ortho1", "ortho2", "ortho3", "ortho4", "r1", "r2", "r3", "r4", "length"])
	l, o1, o2, o3, o4=pred
	for i in range(l.shape[0]):
		c=l[i]*o1[i]
		c=c/(np.linalg.norm(l[i])*np.linalg.norm(o1[i]))
		c1=c.sum()
		
		c=l[i]*o2[i]
		c=c/(np.linalg.norm(l[i])*np.linalg.norm(o2[i]))
		c2=c.sum()

		c=l[i]*o3[i]
		c=c/(np.linalg.norm(l[i])*np.linalg.norm(o3[i]))
		c3=c.sum()

		c=l[i]*o4[i]
		c=c/(np.linalg.norm(l[i])*np.linalg.norm(o4[i]))
		c4=c.sum()

		c=[c1, c2, c3, c4]
		c=[float(i)/sum(c) for i in c]

		resR=[]
		sr=sReal[i].tolist(); sr=[e for e in sr if e!=4]
		if len(sr)<=0:
			sr=[-1]
		e4=[1 for e in sr if e==3]; resR.append(sum(e4)*1.0/len(sr))
		e3=[1 for e in sr if e==2]; resR.append(sum(e3)*1.0/len(sr))
		e2=[1 for e in sr if e==1]; resR.append(sum(e2)*1.0/len(sr))
		e1=[1 for e in sr if e==0]; resR.append(sum(e1)*1.0/len(sr))

		df.loc[len(df)]=c+resR+[len(sr)]
		#print (c+resR)

	#df=df[ ( (df["length"]>20) ) ]
	df.to_csv(saveDf, index=False)
	print (df.shape)			
	return df




