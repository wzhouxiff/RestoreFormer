# # ### Good
exp_name='RestoreFormer'

root_path='experiments'
out_root_path='results'
align_test_path='data/test'
tag='test'

outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir $outdir
fi

python -u scripts/test.py \
--outdir $outdir \
-r $root_path'/'$exp_name'/last.ckpt' \
-c 'configs/RestoreFormer.yaml' \
--test_path $align_test_path \
--aligned

