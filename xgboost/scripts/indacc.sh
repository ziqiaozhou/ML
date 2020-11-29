prefix=$1
datafile=`ls $prefix*.csv`
python3 rules2dag.py  --data=$datafile --sort=$2 --mode=indacc\
 --dev_ratio=$3 $prefix.rule.txt 
