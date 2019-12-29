import numpy as np
from sklearn.feature_selection import RFE, VarianceThreshold,SelectKBest, chi2
from sklearn.svm import SVR
from IPython import embed

def prepare_data(samedata_file,diffdata_file):
    if diffdata_file:
        same_data=np.genfromtxt(samedata_file,filling_values=2,delimiter=",")
        diff_data=np.genfromtxt(diffdata_file,filling_values=2,delimiter=",")
        x=np.concatenate((same_data,diff_data))
        y=np.concatenate((np.zeros((same_data.shape[0],1)),np.ones((diff_data.shape[0],1))))
        now = datetime.now()
        np.savez(now.strftime("%m_%D_%H_%M_%S"),X,Y)
    else:
        data=np.load(samedata_file)
        x=data["arr_0"]
        y=data["arr_1"]
    d=x.shape[-1]
    split_index=np.unique(np.where(x==2)[1])
    assert(split_index.shape[0]>2)
    cs=range(0,split_index[0])
    I=range((split_index[0]+1),split_index[1])
    IAlt=range((split_index[1]+1),split_index[2])
    s=range((split_index[2]+1),split_index[3])
    sAlt=range((split_index[3]+1),split_index[4])
    symbols={"s":s,"salt":sAlt,"c":cs,"I":I,"Ialt":IAlt}
    M={}
    count={"s":0,"salt":0,"c":0,"I":0,"Ialt":0,"none":0}
    for label in symbols:
        var=symbols[label]
        for i in var:
            name="%s_%d"%(label,count[label])
            M[i]=name
            count[label]=count[label]+1
    cols=[]
    vtype={}
    symbol_vars={"s":[],"salt":[],"c":[],"I":[],"Ialt":[]}
    select=VarianceThreshold(0.00001)
    #np.random.shuffle(x)
    #np.random.shuffle(y)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 5, step=1)
    newX=select.fit_transform(x, y)
    #select=RFE(chi2, k=20)
    index=np.where(select.get_support())[0]
    for offset in range(index.shape[0]):
        i=index[offset]
        if i not in M:
            label="none"
            name=label
        else:
            name=M[i]
            label=name.split("_")[0]
            symbol_vars[label].append(offset)
        cols.append(name)

    return (x[:,index],y,cols,symbol_vars)
