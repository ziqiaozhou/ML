import argparse
from skrules import SkopeRules
from prepare_data import prepare_data
from IPython import embed
from sklearn.utils import class_weight,shuffle
import xgboost
import pandas as pd
import numpy as np
import re
from xgboost_rules import xgbtree_rule_perf,xgbtree_to_rules,tocnffile,simplify_rules
"""
data: pandas.dataframe
"""
class LeakageLearner:
    def __init__(self,args):
        self.args=args
        self.cnffile="%s.cnf"%self.args.outname
        self.modelfile="%s.model.txt"%self.args.outname
        self.rulefile="%s.rule.txt"%self.args.outname
        self.attacker_cnffile="%s_attacker.cnf"%self.args.outname
        self.attacker_modelfile="%s_attacker.model.txt"%self.args.outname
        self.attacker_rulefile="%s_attacker.rule.txt"%self.args.outname
    def saverules(self,rules,filename):
        with open(filename,'w') as f:
            f.write("rule,precision,recall,ntree\n")
            for r,scores in rules:
                f.write("%s,%.2f,%.2f,%d\n"%(r,scores[0],scores[1],scores[2]))
        with open(filename+".tex",'w') as f:
            f.write("rule&precision&recall&ntree\\\\\n")
            for r,scores in rules:
                r=r.replace("and","\\wedge")
                r=r.replace(">= 1","= 1")
                r=r.replace("< 1","= 0")
                r=re.sub(r'([a-zA-Z]*)_([0-9]*)',r'\1_{\2}',r)
                f.write("$%s$&%.2f&%.2f&%d\\\\\n"%(r,scores[0],scores[1],scores[2]))
    def loadrules(self,filename):
        pdrules=pd.read_csv(filename)
        return pdrules

    def train(self,X,y,feature_names,symbol_vars):
    #model = xgboost.XGBClassifier(max_depth=7, n_estimators=10)
        y=(y==0)
        X,y=shuffle(X,y)
        #class_w=class_weight.compute_class_weight("balanced",np.unique(y),y)
        sample_weight=class_weight.compute_sample_weight("balanced",y)
        train_size=int(0.8*X.shape[0])
        train_index=range(train_size)
        eval_index=range(train_size,X.shape[0])
        pddata=pd.DataFrame(X[eval_index,:],columns=feature_names)
        pddata.insert(0, 'Y', y[eval_index]==1)
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
        attacker_learner_constraints=feature_combination.copy()
        #feature_combination.append([d-1])
        leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["salt"])
        leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars['Ialt'])
        leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["salt"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["Ialt"]+symbol_vars["c"])

        attacker_learner_constraints.append(symbol_vars["s"]+symbol_vars["salt"])
        attacker_learner_constraints.append(symbol_vars["I"]+symbol_vars['Ialt'])
        attacker_learner_constraints.append(symbol_vars["I"]+symbol_vars["c"])
        attacker_learner_constraints.append(symbol_vars["Ialt"]+symbol_vars["c"])
        params={'max_depth': 8,
                'eta': 1,
                'nthread':8}
        params['objective'] = 'binary:logistic'
        params['max_leaves']=100
        params['tree_method']='exact'
        params['interaction_constraints']=leakage_learner_constraints
        #params['booster']='dart'
        #params['sample_type']='weighted'
        #params['rate_drop']=0.1
        #params['skip_drop']=0.5
        #params['eval_metric']='error'
        #params['scale_pos_weight']=np.where(y==0)[0].shape[0]/np.where(y==1)[0].shape[0]
        model=xgboost.train(
                    params = params,
                    dtrain=data,
                    num_boost_round=5,
                )
        model.dump_model(self.modelfile, with_stats=True)
        clf = SkopeRules(max_depth_duplication=8,
                     precision_min=0.7,
                     recall_min=0.01,
                     verbose=1,
                     feature_names=feature_names)
        clf.fit_xgbmodel(pddata, model)
        clf.rules_.sort(key=lambda x: x[1],reverse=True)
        var_sizes=[32, 116, 0, 5, 5]
        cnf=tocnffile(var_sizes,~simplify_rules(clf.rules_),self.cnffile)
        embed()
        self.saverules(clf.rules_,self.rulefile)
        params['interaction_constraints']=attacker_learner_constraints
        attacker_model=xgboost.train(
            params = params,
            dtrain=data,
            num_boost_round=5,
        )
        model.dump_model(self.attacker_modelfile, with_stats=True)
        attacker_clf = SkopeRules(max_depth_duplication=8,
                     precision_min=0.7,
                     recall_min=0.01,
                     verbose=1,
                     feature_names=feature_names)
        attacker_clf.fit_xgbmodel(pddata, attacker_model)
        attacker_clf.rules_.sort(key=lambda x: x[1],reverse=True)
        attacker_cnf=tocnffile(var_sizes,~simplify_rules(attacker_clf.rules_),self.attacker_cnffile)
        self.saverules(attacker_clf.rules_,self.attacker_rulefile)
        embed()



    def main(self,samedata_file,diffdata_file):
        x,y,feature_names,symbol_vars=prepare_data(samedata_file,diffdata_file)
        #embed()
        self.train(x,y,feature_names,symbol_vars)

    def run(self):
        if len(args.files)==1:
            self.main(args.files[0],None)
        else:
            self.main(args.files[0],args.files[1])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='match symbol')
    parser.add_argument('files',metavar='files',type=str,nargs="+",help='same file, diff file')
    parser.add_argument('--outname',type=str,default="xgboost_1",help='outname')
    args=parser.parse_args()
    learner=LeakageLearner(args)
    learner.run()
