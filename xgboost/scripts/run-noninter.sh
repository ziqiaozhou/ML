dir=$1
name=$2-$3
python3 train_bits.py $dir/diff.csv $dir/same.csv  --outname=$dir/$name --symbol=$dir/symbol.txt --ntrees=64 --depth=8 --nlinear=$3

