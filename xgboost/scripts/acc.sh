prefix=$1
datafile=`ls $prefix*.csv`
python3 rules2dag.py  --data=$datafile --sort=$2 $prefix.rule.txt 
