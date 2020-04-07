import argparse
from skrules import SkopeRules
from prepare_data import prepare_data
from IPython import embed
from sklearn.utils import class_weight,shuffle
import xgboost
import pandas as pd
import numpy as np
import re
from skrules.xgboost_rules import xgbtree_rule_perf,xgbtree_to_rules,tocnffile,simplify_rules,tosymrule
from sympy import simplify
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
    def trim_rule(rule_score):
        rule=rule_score[0]
        conds=rule.split(" and ")
        score=rule_score[1]
        embed()
        newcond=set(conds)
        for cond in conds:
            newcond.remove(cond)
            newrule=" and ".join(list(newcond))
            new_score=xgbtree_rule_perf(str(newrule),self.pddata,self.pddata['Y'])
            if new_score[0]<score[0]*0.95 and new_score[1]<score[1]*2:
                newcond.add(cond)
        newscore=xgbtree_rule_perf(str(newrule),self.pddata,self.pddata['Y'])
        return [" and ".join(list(newcond)),new_score]
 

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

    def train(self,X,y,feature_names,symbol_vars):
    #model = xgboost.XGBClassifier(max_depth=7, n_estimators=10)
        y=(y==self.args.label)
        X,y=shuffle(X,y)
        #class_w=class_weight.compute_class_weight("balanced",np.unique(y),y)
        sample_weight=class_weight.compute_sample_weight("balanced",y)
        train_size=int(0.7*X.shape[0])
        train_index=range(train_size)
        eval_index=range(train_size,X.shape[0])
        self.pddata=pd.DataFrame(X[eval_index,:],columns=feature_names)
        self.pddata.insert(0, 'Y', y[eval_index]==1)
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
        #feature_combination.copy()
        #feature_combination.append([d-1])
        leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["salt"])
        leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars['Ialt'])
        leakage_learner_constraints.append(symbol_vars["s"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["salt"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["I"]+symbol_vars["c"])
        leakage_learner_constraints.append(symbol_vars["Ialt"]+symbol_vars["c"])

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
        if self.args.debug:
            embed()
        model=xgboost.train(
                    params = params,
                    dtrain=data,
                    num_boost_round=10,
                )
        model.dump_model(self.modelfile, with_stats=True)
        clf = SkopeRules(max_depth_duplication=self.args.depth,
                     precision_min=0.6,
                     recall_min=0.01,
                     verbose=1,
                     feature_names=feature_names)
        clf.fit_xgbmodel(self.pddata, model)
        clf.rules_.sort(key=lambda x: x[1],reverse=True)
        sym_vars=symbol_vars
        var_sizes=[len(sym_vars['c']), len(sym_vars['I']), len(sym_vars['Ialt']), len(sym_vars['s']), len(sym_vars['salt'])]
        allr1,allr=simplify_rules(clf.rules_)
        cnf=tocnffile(var_sizes,~allr1,self.cnffile)
        allrscore=xgbtree_rule_perf(str(allr1),self.pddata,self.pddata['Y'])
        print("all r=",simplify(~allr),allrscore)
        self.saverules(clf.rules_,[simplify(allr),allrscore],self.rulefile)
        if self.args.debug:
            embed()
        params['interaction_constraints']=attacker_learner_constraints
        attacker_model=xgboost.train(
            params = params,
            dtrain=data,
            num_boost_round=10,
        )

        model.dump_model(self.attacker_modelfile, with_stats=True)
        attacker_clf = SkopeRules(max_depth_duplication=8,
                     precision_min=0.7,
                     recall_min=0.01,
                     verbose=1,
                     feature_names=feature_names)
        attacker_clf.fit_xgbmodel(self.pddata, attacker_model)
        attacker_clf.rules_.sort(key=lambda x: x[1],reverse=True)
        allr1,allr=simplify_rules(attacker_clf.rules_)
        attacker_cnf=tocnffile(var_sizes,~allr1,self.attacker_cnffile)
        allrscore=xgbtree_rule_perf(str(simplify(allr1)),self.pddata,self.pddata['Y'])
        self.saverules(attacker_clf.rules_,[simplify(allr),allrscore],self.attacker_rulefile)



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
    
    parser.add_argument('--debug',type=bool,default=False,help='decision rule depth')
    parser.add_argument('--depth',type=int,default=10,help='decision rule depth')
    parser.add_argument('--outname',type=str,default="xgboost_1",help='outname')
    parser.add_argument('--label',type=int,default=0,help='label value')
    args=parser.parse_args()
    learner=LeakageLearner(args)
    learner.run()
