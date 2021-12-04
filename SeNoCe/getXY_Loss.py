import sys
import pandas as pd
import numpy as np
import pickle 
from ast import literal_eval as le

pin="../../Data/emRumEval/"
pout="repo/"
trainFinal="repo/trainFinal.csv"
testFinal="repo/testFinal.csv"

kwargs={"dict":pin+"tweetId2Embed_Dict.pkl", "pad":pin+"rumEvalPad.csv", "tableDf":pin+"rumEvalTable.csv"}

#pd.Series(df.Letter.values,index=df.Position).to_dict()
#d=dfD.set_index("tweetId").to_dict()["padded_docs"]
#dfD=dfD.drop_duplicates(subset="tweetId", keep="first")

def xyRet():
	dtr, dts=pd.read_csv(trainFinal), pd.read_csv(testFinal)
	for i in range(10):
		dtr=dtr.sample(frac=1)
		dts=dts.sample(frac=1)
	#dtr=dtr[:1488]
	dtr["srcEmb"]=dtr["srcEmb"].apply(lambda x: le(x))
	dtr["branchEmb"]=dtr["branchEmb"].apply(lambda x: le(x))
	dtr["veracity"]=dtr["veracity"].apply(lambda x: le(x))

	dts["srcEmb"]=dts["srcEmb"].apply(lambda x: le(x))
	dts["branchEmb"]=dts["branchEmb"].apply(lambda x: le(x))
	dts["veracity"]=dts["veracity"].apply(lambda x: le(x))

	for i in range(1000):
		dtr=dtr.sample(frac=1)
	
	##for training
	sp=dtr.shape
	srcEmb=np.zeros((sp[0], len(dtr["srcEmb"].tolist()[0])))
	branchEmb=np.zeros((sp[0], len(dtr["branchEmb"].tolist()[0])))
	veracity=np.zeros((sp[0], len(dtr["veracity"].tolist()[0])))
	for i, row in dtr.iterrows():
		srcEmb[i, :]=row["srcEmb"]
		branchEmb[i, :]=row["branchEmb"]
		veracity[i, :]=row["veracity"]

	Xtr=[  srcEmb, branchEmb, branchEmb ]
	print ([x.shape for x in Xtr])
	orthoL=[np.zeros((sp[0], 1)) for i in range(6)]
	Ytr=[ veracity, veracity ]+orthoL	
	print ([x.shape for x in Ytr])


	##for testing
	sp=dts.shape
	srcEmb=np.zeros((sp[0], len(dts["srcEmb"].tolist()[0])))
	branchEmb=np.zeros((sp[0], len(dts["branchEmb"].tolist()[0])))
	veracity=np.zeros((sp[0], len(dts["veracity"].tolist()[0])))
	for i, row in dts.iterrows():
		srcEmb[i, :]=row["srcEmb"]
		branchEmb[i, :]=row["branchEmb"]
		veracity[i, :]=row["veracity"]

	Xts=[  srcEmb, branchEmb, branchEmb ]
	#print ([x.shape for x in Xts])
	orthoL=[np.zeros((sp[0], 1)) for i in range(6)]
	Yts=[ veracity, veracity ]+orthoL	
	#print ([x.shape for x in Yts])

	return Xtr, Ytr, Xts, Yts

def getXYArr(**kwargs):
	dfTable=kwargs["dfTable"]
	dicT=kwargs["dicT"]
	mx=kwargs["mx"]
	p3=kwargs["p3"]

	for i in range(1000):
		dfTable=dfTable.sample(frac=1)
	dfTable=dfTable.reset_index(drop=True)

	##managing stances labels
	dfTable['fold_stance_labels']=dfTable['fold_stance_labels'].apply(lambda x:le(x))
	
	if "saveDf" in kwargs:
		dfTable.to_csv(pt+kwargs["saveDf"], index=False)

	N, _=dfTable.shape

	col=dfTable.columns.tolist()#; print (col); input("enter"); # ['topics', 'source_id', 'branches', 'rnr_labels', 'fold_stance_labels', 'Veracitylabels']
	dfXY=pd.DataFrame(columns=col+["srcEmb", "branchEmb", "veracity"])	

	for i, row in dfTable.iterrows():
		lis=list(row)#; print (lis); input("enter")

		if len(lis[1])>mx:
			lis[1]=lis[1][:mx]
		
		tempBr=[]
		for ib in range(mx):
			if ib <= mx:
				try:
					tempBr.append( np.mean( dicT[ int( lis[1][ib] ) ] ) )
				except:
					tempBr.append(0)
			else:
				tempBr.append(0)
		tempBr=[tempBr]	

		##labels for veracity
		vsL=[]
		if lis[5]==2:
			vsL=[[1, 0, 0]]
		elif lis[5]==1:
			vsL=[[0, 1, 0]]
		else:
			vsL=[[0, 0, 1]]

		lisAdd=[dicT[lis[0]]]+tempBr+vsL
		dfXY.loc[len(dfXY)]=lis+lisAdd
		#print (dfXY.iloc[-1].tolist())
	dfXY["source_id"]=dfXY["source_id"].apply(lambda x: str(x)+"_")
	#print ("shape of dfXY ", dfXY.shape)
	dfXY[:p3].to_csv(trainFinal, index=False); dfXY[p3:].to_csv(testFinal, index=False)	
	return  

def mixLeast(df, p):
	#print ("**********actual rnr ", df[( (df["rnr_labels"]==True) )].shape[0], df[( (df["rnr_labels"]==False) )].shape[0]); input("enter a number")
	dff=pd.DataFrame()
	#nr=0.8; r=0.2
	#while max(nr, r)>=1.5*min(nr, r):
	df=df.sample(frac=1)

	sT=df[( (df["Veracitylabels"]==0) )].shape[0]
	sF=df[( (df["Veracitylabels"]==1) )].shape[0]
	sU=df[( (df["Veracitylabels"]==2) )].shape[0]
	mn=min(sT, sF, sU)
	dfTr, dfTs=df[( (df["Veracitylabels"]==0) )][:int(mn*p)], df[( (df["Veracitylabels"]==0) )][int(mn*p):]
	dfFr, dfFs=df[( (df["Veracitylabels"]==1) )][:int(mn*p)], df[( (df["Veracitylabels"]==1) )][int(mn*p):]
	dfUr, dfUs=df[( (df["Veracitylabels"]==2) )][:int(mn*p)], df[( (df["Veracitylabels"]==2) )][int(mn*p):]

	dff=pd.concat([dfTr, dfFr, dfUr, dfTs, dfFs, dfUs])	
	return dff, 3*int(mn*p)

def getSamples(**kwargs):
	Dict=kwargs["dict"]
	pad=kwargs["pad"]
	tableDf=kwargs["tableDf"]

	f=open(Dict, "rb")
	dicT=pickle.load(f)
	f.close()
	dlis=list(dicT.keys())

	dfTable=pd.read_csv(tableDf)

	dfTable["source_id"]=dfTable["source_id"].apply(lambda x: int(x[:-1]))
	dfTable["branches"]=dfTable["branches"].apply(lambda x:le(x))

	#getting maximum length of each tweets
	tempL=[len(x) for x in dfTable["branches"]]; mx=max(tempL)#; mx=50
	print (dfTable.columns.tolist(), dfTable.shape)
	print ("maximum length in a branch ", mx)

	dff, p3=mixLeast(dfTable, 0.8)

	params={"dfTable":dff, "dicT":dicT, "mx":mx, "p3":p3}
	getXYArr(**params)	
	return 


#getSamples(**kwargs)
xyRet()

