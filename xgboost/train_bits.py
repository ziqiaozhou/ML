import argparse
from skrules import SkopeRules
from prepare_data import prepare_data
from IPython import embed
from sklearn.utils import class_weight,shuffle
import xgboost
import pandas as pd
import numpy as np
import re
print("hi")
from skrules.xgboost_rules import xgbtree_rule_perf,xgbtree_to_rules,tocnffile,simplify_rules,tosymrule
from sympy import simplify
print("hi")
from customLinear import LinearFeature
#import matplotlib.pyplot as plt
print("hi")
"""
data: pandas.dataframe
"""

def toLatex(rule_scores,filename):
    def ruleToLatex(rule):
        rule=re.sub("L_([0-9]*) ",r"\\linearFeature{\1}",rule)
        rule=re.sub("diff_s_([0-9]*) ",r"\\diffFeature{\1}",rule)
        rule=re.sub("s_([0-9]*) ",r"\\SecFn{}[\1]",rule)
        rule=re.sub("c_([0-9]*) ",r"\\ACIFn{}[\1]",rule)
        rule=re.sub("I_([0-9]*) ",r"\\AIIFn{}[\1]",rule)
        rule=re.sub("salt_([0-9]*)",r"\\SecFnAlt{}[\1]",rule)
        rule=re.sub(">=",r"\\ge",rule)
        conds=rule.split(" and ")
        conds.sort(key=lambda x: x[1],reverse=True)
        rule=" \\wedge ".join(conds)
        #rule=re.sub("and",r"\\wedge",rule)
        return rule
    i=0
    ret=[]
    for rule_score in rule_scores:
        rule=rule_score[0]
        score=rule_score[1]
        display="$\\interferenceRule_{%d}$&$%s$&%.2f&%.2f\\\\"%(i,ruleToLatex(rule),score[0],score[1])
        print(display)
        ret.append(display)
        i=i+1
    with open(filename,'w') as f:
        f.write("\n".join(ret))

def trim_rule(rule_score,pddata,thres=0.05):
	rule=rule_score[0]
	conds=rule.split(" and ")
	score=rule_score[1]
	newcond=set(conds)
	for cond in conds:
		newcond.remove(cond)
		newrule=" and ".join(list(newcond))
		new_score=xgbtree_rule_perf(str(newrule),pddata,pddata['Y'])
		if new_score[0]<score[0]*(1-thres):
			newcond.add(cond)
	newrule=" and ".join(list(newcond))
	newscore=xgbtree_rule_perf(str(newrule),pddata,pddata['Y'])
	return [newrule,newscore]

class LeakageLearner:
    def __init__(self,args):
        self.args=args
        self.cnffile="%s.cnf"%self.args.outname
        self.modelfile="%s.model.txt"%self.args.outname
        self.rulefile="%s.rule.txt"%self.args.outname
        self.attacker_cnffile="%s_attacker.cnf"%self.args.outname
        self.attacker_modelfile="%s_attacker.model.txt"%self.args.outname
        self.attacker_rulefile="%s_attacker.rule.txt"%self.args.outname
        self.rule_latex="%s_rule.tex"%self.args.outname

    def trim_rule(self,rule_score):
        rule=rule_score[0]
        conds=rule.split(" and ")
        score=rule_score[1]
        newcond=set(conds)
        for cond in conds:
            newcond.remove(cond)
            newrule=" and ".join(list(newcond))
            new_score=xgbtree_rule_perf(str(newrule),self.pddata,self.pddata['Y'])
            if new_score[0]<score[0]*0.97:
                newcond.add(cond)
        newrule=" and ".join(list(newcond))
        newscore=xgbtree_rule_perf(str(newrule),self.pddata,self.pddata['Y'])
        return [newrule,newscore]
    def saverules(self,rules,allrstat,filename):
        allscores=allrstat[1]
        with open(filename,'w') as f:
            f.write("rule,precision,recall,ntree\n")
            for r,scores in rules:
                    f.write("%s,%.2f,%.2f,%d\n"%(r,scores[0],scores[1],scores[2]))
            r=allrstat[0]
            f.write("%s,%.2f,%.2f,--\n"%(r,allscores[0],allscores[1]))
        with open(filename+".tex",'w') as f:
            f.write("rule&precision&recall&ntree\\\\\n")
            for r,scores in rules:
                r=r.replace("and","\\wedge")
                r=r.replace(">= 1","= 1")
                r=r.replace("< 1","= 0")
                r=re.sub(r'([a-zA-Z]*)_([0-9]*)',r'\1_{\2}',r)
                f.write("$%s$&%.2f&%.2f&%d\\\\\n"%(r,scores[0],scores[1],scores[2]))

            f.write("$%s$&%.2f&%.2f&--\\\\\n"%(str(allrstat[0]),allscores[0],allscores[1]))
    def loadrules(self,filename):
        pdrules=pd.read_csv(filename)
        return pdrules

    def train(self,feature_names,symbol_vars):
    #model = xgboost.XGBClassifier(max_depth=7, n_estimators=10)
            #class_w=class_weight.compute_class_weight("balanced",np.unique(y),y)
        sample_weight=class_weight.compute_sample_weight("balanced",self.pddata['Y'])
        self.pddata['Y']=(self.pddata['Y']==self.args.label)
        X=self.pddata.iloc[:,1:].to_numpy()
        y=self.pddata['Y']
        data=xgboost.DMatrix(data=X,
                            label=y,
                            feature_names=feature_names,
                            feature_types=['int']*X.shape[-1],
                            weight=sample_weight)
        d=X.shape[-1]
        feature_combination=[]
        for sym in symbol_vars:
                print(sym)
                if len(symbol_vars[sym])>0:
                        feature_combination.append(symbol_vars[sym])
        leakage_learner_constraints=feature_combination.copy()
        attacker_learner_constraints=[]
        """
            leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["salt"])
            leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars['Ialt'])
            leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["c"])
            leakage_learner_constraints.append(symbol_vars["salt"]+symbol_vars["c"])
            leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars["c"])
            leakage_learner_constraints.append(symbol_vars["Ialt"]+symbol_vars["c"])
            attacker_learner_constraints.append(symbol_vars["I"]+symbol_vars['Ialt'])
            attacker_learner_constraints.append(symbol_vars["I"]+symbol_vars["c"])
            attacker_learner_constraints.append(symbol_vars["Ialt"]+symbol_vars["c"])
        """
        params={'max_depth': self.args.depth+2,
                'eta': 1,
                'nthread':8}
        params['objective'] = 'binary:logistic'
        params['max_leaves']=100
        #params['tree_method']='hist'
        params['grow_policy']='lossguide'
        #params['subsample']=0.9
        #params['gamma']=0.01 #Minimum loss reduction
        #params['interaction_constraints']=leakage_learner_constraints
        #params['booster']='dart'
        #params['sample_type']='weighted'
        #params['rate_drop']=0.1
        #params['skip_drop']=0.5
        #params['eval_metric']='error'
        #params['scale_pos_weight']=np.where(y==0)[0].shape[0]/np.where(y==1)[0].shape[0]
        if self.args.debug:
            embed()
        model=xgboost.train(params = params,
                            dtrain=data,
                            num_boost_round=self.args.ntrees,
                        )
        model.dump_model(self.modelfile, with_stats=True)
        clf = SkopeRules(max_depth_duplication=self.args.depth,
                                 precision_min=0.6,
                                 recall_min=0.01,
                                 verbose=1,
                                 feature_names=feature_names)
        evaldata=self.pddata.sample(frac=0.3, replace=True)
        evaldata=evaldata.reset_index()
        eval_sample_weight=class_weight.compute_sample_weight("balanced",evaldata['Y'])
        clf.fit_xgbmodel(evaldata, model, eval_sample_weight)
        clf.rules_.sort(key=lambda x: x[1],reverse=True)
        rules={}
        for i in range(len(clf.rules_)):
            r=trim_rule(clf.rules_[i],self.pddata)
            rules[r[0]]=r[1]
        rulelist=[]
        for r in rules:
            rulelist.append([r,rules[r]])
        rulelist.sort(key=lambda x: x[1],reverse=True)
        usedLinear={}
        toLatex(rulelist,self.rule_latex)
        for lname in self.linear:
            if any(lname in r[0] for r in rulelist ):
                usedLinear[lname]=self.linear[lname]
                print("%s=%s"%(lname,usedLinear[lname][0]))

        sym_vars=symbol_vars
        var_sizes=[len(sym_vars['c']), len(sym_vars['I']), len(sym_vars['Ialt']), len(sym_vars['s']), len(sym_vars['salt'])]
        allr1,allr=simplify_rules(clf.rules_)
        embed()
        #cnf=tocnffile(var_sizes,allr1,self.cnffile)
        allrscore=xgbtree_rule_perf(str(allr1),self.pddata,self.pddata['Y'])
        print("all r=",simplify(~allr),allrscore)
        self.saverules(clf.rules_,[simplify(allr),allrscore],self.rulefile)
        if self.args.debug:
            embed()

    def localLearner(self,pddata,symbol_vars):
        continuous={}
        #continuous['s']=[(0,8,"secret")]
        #continuous['salt']=[(0,8,"secret'")]
        #continuous['c']=[(0,8,"offset")]
        #continuous['I']=[(0,8,"arr1_size")]
        intCols=['Y']
        with open(self.args.symbol) as symbolfile:
            for line in symbolfile:
                line.replace("\n","")
                w=line.split()
                if w[0] not in continuous:
                    continuous[w[0]]=[]
                continuous[w[0]].append((int(w[1]),int(w[2]),w[3]))
        ncols=pddata.shape[1]
        for xtype in continuous:
            x_val=0
            base=1;
            for x_range in continuous[xtype]:
                x_name=x_range[2]
                start=x_range[0]
                end=x_range[1]+start
                valid=True
                for i in range(start,end):
                    name="%s_%d"%(xtype,i)
                    if name not in pddata.columns:
                        x_val=0
                        valid=False
                        break
                    x_val=x_val+pddata[name]*base
                    base=base*2
                symbol_vars[xtype].append(x_name)
                if x_name not in pddata.columns and valid:
                    print("add",x_name)
                    pddata.insert(ncols,x_name, x_val)
                    intCols.append(x_name)
        intData=pddata[intCols]
        lf=LinearFeature()
        lf.fit(intData)
        self.lf=lf
        feature,addeddata=lf.features(0.8)
        return feature,pd.DataFrame(data=addeddata)

    def main(self,samedata_file,diffdata_file):
        X,y,feature_names,symbol_vars=prepare_data(samedata_file,diffdata_file)
        #embed()
        pddata=pd.DataFrame(X,columns=feature_names)
        pddata.insert(0, 'Y',  y)
        linear,addeddata=self.localLearner(pddata,symbol_vars)
        self.linear=linear
        for name in feature_names:
            if "s_" in name:
                name2=name.replace('s_',"salt_")
                val=abs(pddata[name]-pddata[name2])
                addeddata["diff_%s"%name]=val.to_numpy()
        pddata=pddata.join(pd.DataFrame(data=addeddata))
        self.pddata=pddata
        feature_names=pddata.columns[1:]
        self.train(feature_names,symbol_vars)

    def run(self):
        if len(args.files)==1:
            self.main(args.files[0],None)
        else:
            self.main(args.files[0],args.files[1])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='match symbol')
    parser.add_argument('files',metavar='files',type=str,nargs="+",help='same file, diff file')
    parser.add_argument('--debug',type=bool,default=False,help='decision rule depth')
    parser.add_argument('--depth',type=int,default=6,help='decision rule depth')
    parser.add_argument('--ntrees',type=int,default=10,help='decision trees')
    parser.add_argument('--outname',type=str,default="xgboost_1",help='outname')
    parser.add_argument('--symbol',type=str,default="symbol.txt",help='outname')
    parser.add_argument('--label',type=int,default=1,help='label value')
    args=parser.parse_args()
    learner=LeakageLearner(args)
    learner.run()
