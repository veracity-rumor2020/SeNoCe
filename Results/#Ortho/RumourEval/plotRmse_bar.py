import matplotlib.pyplot as plt
import matplotlib 
matplotlib.style.use('ggplot')
import pandas as pd

#import modData as md

font = { 'color': 'black',
        'weight' : 'bold',
        'size'   : 55}
       
fontX = {'color':'black',
         'weight' : 'bold',
	 'size'   : 55
	}

# def getDf():
# 	dff=pd.DataFrame()
# 	df=pd.read_csv("macroF.csv"); col=df.columns.tolist()
# 	dff["#Orthogonal Features"]=df[col[0]]; dff["MacroF(Pheme5)"], dff["MacroF(RumEval)"]=df[col[2]], df[col[4]]
# 	df=pd.read_csv("accuracy.csv"); col=df.columns.tolist()
# 	dff["Acc(Pheme5)"], dff["Acc(RumEval)"]=df[col[2]], df[col[4]]
# 	print (dff)
# 	return dff

# df=getDf()#; input("enter a number")

#df=pd.read_csv("outHA.csv")
#df=pd.read_csv("outHA_IEEE.csv")

df=pd.read_csv("OrthoRumeval_revised.csv")

xLabelString="#OFs" 
yLabelString="Score"

	

ax=df.plot(x="#OFs", color=["blue","green"], kind="bar",  rot=0)#, linewidth=8, style=[":","--","-.", "-^"]) alpha = 0.6,

#ax.set_xlabel(None)
ax.set_xlabel(xLabelString,**font)
ax.set_ylabel(yLabelString,**font)

ax.xaxis.set_tick_params(labelsize=60, colors='black')
ax.yaxis.set_tick_params(labelsize=60, colors='black')

#ax.grid(color="black", linewidth=1)
ax.set_facecolor("w")

#for p in ax.patches:
#	ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), size=40, weight="bold")

for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2)
	ax.spines[axis].set_color("black")

#plt.legend(bbox_to_anchor=(0.399, 1.11), fontsize=15, loc="upper right", ncol=2)
plt.legend(fontsize=55, loc="lower left", ncol=2)
#plt.title(title, **font)

#plt.xticks(rotation=90)
plt.savefig("Ortho_rumEval.eps")
plt.show()

print ("done")
