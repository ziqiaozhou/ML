from sklearn.svm import LinearSVC, SVC
from IPython import embed
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
X = []
Y = []
for i in np.linspace(1, 100,30):
    for j in np.linspace(-100,100,30):
        for k in np.linspace(-100,100,30):
            x1=i
            x2=x1+j
            x3=x1+k
            if (x1<x2 and x1>x3) or (x1>=x2 and x1<=x3):
                y=1
            else:
                y=0
            X=X+[[x1,x2,x3]]
            Y=Y+[y]
baselabel=2
X=np.array(X)
Y=np.array(Y)

"""
newX=[]
newY=[]
for y in [0,1]:
    #embed()
    tmpX=X[Y==y]
    print (len(tmpX))
    clustering = DBSCAN(eps=3, min_samples=10).fit(tmpX)
    newX=newX+list(tmpX)
    labels=clustering.labels_+baselabel
    nlabel=len(np.unique(labels))
    newY=newY+list(labels)
    print("clustering to nlabel=")
    print(nlabel)
    baselabel=baselabel+nlabel
"""
#data=np.array(X)
#data=np.append(data,np.array(Y).reshape([-1,1]),1)
class SVCTree:
    def getSV(i):
        if i>-1:
            return clf.support_vector_[:,i]
        else:
            return clf.coef0
    def getVarName(i):
        if i<0:
            return ''
        return 'X'+str(i)
    def poly_str(self,clf):
        feature_map={}
        nfeature=clf.support_vector_.shape[1]
        for i in range(-1,nfeature):
            feature_map[i]={}
            for j in range(i,nfeature):
                feature_map[i][j]=0
        if clf.degree==2:
            for index1 in range(-1,nfeature):
                for index2 in  range(-1,nfeature):
                    i=min(index1,index2)
                    j=max(index1,index2)
                    svi=self.getSV(i);
                    svj=self.getSV(j);
                    feature_map[i][j]=feature_map[i][j]+np.sum(svi*svj)
        for i in feature_map:
            for j in range(i,nfeature):
                if feature_map[i][j]<0:
                    s=s+'-'
                else:
                    s=s+'+'
                s=s+'{:.1f}'.format(abs(feature_map[i][j]))+self.getVarName(i)+'*'+self.getVarName(j)
        return s
    def __init__(self,nodeid,level,feature_combination,
            min_node_size=100,max_level=10,tol=0.000001,model=LinearSVC):
        self.left=None
        self.right=None
        self.clf=None
        self.count=0
        self.min_node_size=min_node_size
        self.nodeid=nodeid
        self.level=level;
        self.feature_combination=feature_combination
        self.score=0
        self.base_score=0
        self.tol=tol
        self.sampleSize=0
        self.model=model
    def predict(self,X):
        return 0

    def node2str(self):
        s=""
        if self.clf.kernel=='poly':
            s=s+self.poly_str()
        elif self.clf!=None:
            coef=self.clf.coef_[0]
            small=np.min(np.abs(coef))
            coef=coef/small
            intercept=self.clf.intercept_[0]/small
            s=s+str("{:.1f}".format(coef[0]))+"*X"+str(feature_combination[0])
            i=0
            for w in coef[1:]:
                i=i+1
                j=self.feature[i]
                if w<0:
                    s=s+'-'+"{:.1f}".format(-w)+'*X'+str(j)
                elif w>0:
                    s=s+'+'+"{:.1f}".format(w)+'*X'+str(j)
            if intercept<0:
                s=s+'-'+"{:.1f}".format(-intercept)
            if intercept>0:
                s=s+'+'+"{:.1f}".format(intercept)
        s=s+ '\nbase='+"{:.2f}".format(self.base_score)+'\nscore='+"{:.2f}".format(self.score)+"\nsample="+str(self.sampleSize)
        return s

    def build_svc(self,X,Y,level,base_acc=0):
        best_clf=None

        self.base_score=best_score=base_acc
        best_pY=np.array([])
        self.score=base_acc
        self.sampleSize=X.shape[0]
        if len(np.unique(Y))==1:
            self.score=1.0
            return 1;
        if len(Y)<self.min_node_size or base_acc>1.0-self.tol:
            return 1;
        for comb in feature_combination:
            if self.model !='sgd':
                clf = SVC(tol=1e-4,kernel=self.model,degree=2,gamma='auto')
            else:
                clf = SGDClassifier(loss='hinge',learning_rate='adaptive',eta0=0.02,epsilon=0.0001,tol=0.0000001)
            clf.fit(X[:,comb], Y)
            pY=clf.predict(X[:,comb])
            score=metrics.accuracy_score(Y, pY)

            print (comb,score,base_acc)
            if score>best_score:
                best_score=score
                best_clf=clf
                best_pY=pY
                self.feature=comb
        self.clf=best_clf

        if self.clf==None:
            return 1

        self.count=self.count+1
        pY=best_pY
        print pY
        print("Improve accuracy by ",best_score-base_acc)
        self.score=best_score
        if level==0:
            return self.count
        left_acc=metrics.accuracy_score(Y[pY==1],pY[pY==1])
        right_acc=metrics.accuracy_score(Y[pY==0],pY[pY==0])

        self.left= SVCTree(self.nodeid+1,self.level-1,self.feature_combination)
        subcount=1
        subcount=self.left.build_svc(X[pY==1],Y[pY==1],level-1,left_acc)
        self.count=self.count+subcount
        self.right=SVCTree(self.nodeid+self.count,self.level-1,self.feature_combination)
        subcount=self.right.build_svc(X[pY==0],Y[pY==0],level-1,right_acc)
        self.count=self.count+subcount
        return self.count

    def _export_to_dot(self,f):
        node_id=0
        #if self.clf==None:
        #    return
        stack=[[tree,0]]
        exp= self.node2str()
        f.write('%d [label="%s"];\n'
                               % (self.nodeid,exp))
        if self.left!=None:
            f.write("%d -> %d\n [label=False]" % (self.nodeid,self.left.nodeid))
            self.left._export_to_dot(f)
        if self.right!=None:
            f.write("%d -> %d\n [label=True]" % (self.nodeid,self.right.nodeid))
            self.right._export_to_dot(f)


    def export_to_dot(self,filename):
        f=open(filename,'w+')
        f.write("digraph g {\n")
        self._export_to_dot(f)
        f.write("}")
        f.close()
        
feature_combination=[[0,1,2]] #[0],[1],[2],[0,1],[0,2],[1,2],
tree= SVCTree(0,4,feature_combination,model='poly') #SGDClassifier
tree.build_svc(X,Y,4,base_acc=0)
embed()
tree.export_to_dot('test.dot')
embed()
