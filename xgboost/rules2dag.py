from skrules.xgboost_rules import simplify_rules,tosymrule
import argparse
from IPython import embed
from sympy import simplify

class DAG:
  def __init__(self,rulef, outf):
  	self.rules = []
  	first = True
  	with open(rulef) as f:
  		for line in f:
  			if first:
  				first = False
  				continue
  			rule_perf = line.split(",")
  			self.rules.append(tosymrule(rule_perf[0])[0])
  
  def to_dag(self):
  	n = len(self.rules) 
  	for i in range(n):
  		for j in range(n):
  			if i == j:
  				continue
  			r1 = self.rules[i]
  			r2 = self.rules[j]
  			r12 = simplify((~r1)&r2)
  			print(f"{r1},{r2},{r12}")
  			if not r12:
  				embed()

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='match symbol')
  parser.add_argument('rulefile',metavar='files',type=str,help='.rule.txt files')
  parser.add_argument('--outname',type=str,default="out.dag",help='outname')
  args=parser.parse_args()
  dag = DAG(args.rulefile, args.outname)
  dag.to_dag()