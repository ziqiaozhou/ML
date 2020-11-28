import pandas as pd
from IPython import embed
import random
import numpy as np
from customLinear import LinearFeature
from gplearn.genetic import SymbolicClassifier,SymbolicTransformer
from gplearn.functions import make_function
from sklearn.utils.class_weight import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.linear_model import LinearRegression,SGDClassifier
#from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
#from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
def tobits(onerow,val,size,label):
	binval=format(val, '0%db'%size)
	for i in range(size):
		onerow["%s_%d"%(label,i)]=int(binval[size-i-1])
Ssize=4
Csize=4
Isize=4
cols=[]

cols.append("Y")
"""
for i in range(Csize):
	cols.append("c_%d"%i)
for i in range(Isize):
	cols.append("I_%d"%i)
for i in range(Ssize):
	cols.append("s_%d"%i)
for i in range(Ssize):
	cols.append("salt_%d"%i)
"""
cols.append("secret")
cols.append("secretAlt")
cols.append("control")
cols.append("other")

data=pd.DataFrame(columns=cols,dtype='int')
for i in range(4000):
	secret=random.getrandbits(Ssize)
	secretAlt=random.getrandbits(Ssize)
	control=random.getrandbits(Ssize)
	other=random.getrandbits(Ssize)
	if secret == secretAlt:
		continue
	interference=1
	onerow={}
	onerow={"secret":secret,"control":control,"other":other,"secretAlt":secretAlt}
	"""
	tobits(onerow,secret,Ssize,"s")
	tobits(onerow,secretAlt,Ssize,"salt")
	tobits(onerow,control,Csize,"c")
	tobits(onerow,other,Isize,"I")
	"""
	observe=1 if secret > control else 0
	observeAlt=1 if secretAlt > control else 0
	if observe==observeAlt:
		interference=0
	onerow['Y']=interference
	data=data.append(onerow,ignore_index=True)
from svmtree.SVMTree import SVMTree
xdata=data.iloc[:,1:]
y=data.iloc[:,0]
desc=xdata.describe()
MEAN = np.array(desc.T['mean']) 
STD = np.array(desc.T['std'])
NUMERIC_FEATURES=xdata.columns
#normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

#numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
#numeric_columns = [numeric_column]
#numeric_column
#model=tf.keras.experimental.LinearModel(loss=)
from sklearn import preprocessing
lf=LinearFeature()
lf.fit(data)
lf.features()
data_scaled=(data-data.min()).div(data.max()-data.min())

def data2midpoints(data,xdata):
	min_max_scaler = preprocessing.MinMaxScaler()
	#data_scaled = min_max_scaler.fit_transform(data)
	data_scaled=(data-data.min()).div(data.max()-data.min())
	midpoints=pd.DataFrame(columns=xdata.columns)
	for attribute in xdata.columns:
		data.sort_values([attribute])
		z=data_scaled.iloc[1:,:]
		z.index=z.index-1 
		candidates=(z+data_scaled.iloc[0:-1,])/2
		candidates=candidates[candidates["Y"]==0.5]
		midpoints=midpoints.append(candidates,ignore_index=True,sort=False)
	return midpoints

midpoints=data2midpoints(data,xdata)
NStep=40
selected_points=midpoints.sample(NStep)
sample_weight=compute_sample_weight('balanced',y)
embed()
regs=[]
for point in selected_points.to_numpy():
	dis=(data_scaled-point).iloc[:,1:]
	dis=(dis*dis).sum(axis=1)
	subsample_weight=sample_weight*np.where(dis<0.2,1,0)
	reg=SGDClassifier(penalty='elasticnet')
	reg.fit(xdata,y,sample_weight=subsample_weight)
	score=reg.score(xdata,y,sample_weight=subsample_weight)
	regs.append([reg,score,reg.coef_,reg.intercept_,subsample_weight])

ncenterpoints=30
def minlocalloss(point,y_true, y_pred):
	y_true_label=y_true[:,0]
	y_pred_label=y_pred[:,0]
	points=midpoints.sample(ncenterpoints)
	nfeature=len(cols)-1
	loss=tf.keras.losses.binary_crossentropy(y_true_label,y_pred_label)
	distances=tf.keras.backend.sum((y_true-point)[:,1:]*(y_true-point)[:,1:],axis=1)+0.0000001
	return abs(y_true_label-y_pred_label)/(distances)
	#local_idx=tf.keras.backend.less(tf.keras.backend.sum((y_true-point)[:,1:]*(y_true-point)[:,1:],axis=1),0.05*nfeature)
	loss=np.min(loss,tf.keras.losses.binary_crossentropy(tf.boolean_mask(y_true_label,local_idx),tf.boolean_mask(y_pred_label,local_idx)))
	return loss 

def mean_pred(y_true, y_pred):
	y_true_extended=y_pred*y_true+ y_pred*(1-y_true)
	return tf.keras.losses.categorical_crossentropy(y_true_extended,y_pred)
def myacc(y_true, y_pred):
	y_true_extended=y_pred*y_true+ y_pred*(1-y_true)
	return tf.keras.backend.sum(y_true_extended-y_pred)

def mean_pred(y_true, y_pred):
	y_true_extended=y_pred*y_true+ y_pred*(1-y_true)
	return tf.keras.losses.categorical_crossentropy(y_true_extended,y_pred)
inputs = tf.keras.layers.Input(shape=(4,)) 
features =tf.keras.layers.Dense(2,kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005),bias_regularizer=regularizers.l2(1e-4), activation='relu')(inputs)
#outputs = tf.keras.layers.Dense(1)(features)
outputs=tf.keras.layers.Dense(len(cols), activation='sigmoid')(features)
model = tf.keras.Model(inputs=inputs, outputs=features)
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
model.compile(loss=minlocalloss,
	 optimizer='adam',
	 metrics=[minlocalloss for i in range()])  
sample_weight=compute_sample_weight('balanced',y)
model.fit(xdata,data,sample_weight=sample_weight,epochs=100)
embed()
dummy_y=tf.keras.utils.to_categorical(y)
estimator=KerasClassifier(build_fn=model,epochs=100)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, xdata, y, cv=kfold)
model.fit(xdata,y,sample_weight=sample_weight,epochs=100)
embed()
"""
tree=SVMTree()

embed()
tree.fit(X,Y)

"""
lf=LinearFeature()
lf.fit(data)
embed()
X=data.iloc[:,1:].to_numpy()
Y=data.iloc[:,0:1].to_numpy()
Y=Y.reshape([-1])
class_weight=compute_class_weight('balanced',[0,1],Y)
sample_weight=compute_sample_weight('balanced',Y)
#data.to_csv("dummy.csv")
function_set = ['add', 'sub','mul']
est = SymbolicClassifier(function_set=function_set)
est.fit(X,Y,  sample_weight=sample_weight)
y_pred = est.predict(X)

def _mod(x1, x2):
	return np.where(np.abs(x2) > 0,np.mod(x1,x2),0)

def _ite(x1,x2,x3):
	return np.where(x1>0,x2,x3)
ite = make_function(function=_ite,
						name='ite',
						arity=3)
mod = make_function(function=_mod,
						name='mod',
						arity=2)

gp = SymbolicTransformer(generations=20, population_size=2000,
						 hall_of_fame=100, n_components=10,
						 function_set=function_set,
						 p_crossover=0.9,
						 parsimony_coefficient="auto",
						 max_samples=0.9, verbose=1,
						 random_state=0, n_jobs=1)
gp.fit(X,Y, sample_weight=sample_weight)

embed()
