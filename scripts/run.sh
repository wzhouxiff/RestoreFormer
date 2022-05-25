export BASICSR_JIT=True

conf_name='HQ_Dictionary'
# conf_name='RestoreFormer'

gpus='0,1,2,3'

#P: pretrain SL: soft learning
node_n=1

python -u main.py \
--root-path /mnt/lustre/wangzhouxia/Data_t1/checkpoints/Taming/release \
--base 'configs/'$conf_name'.yaml' \
-t True \
--postfix $conf_name \
--gpus $gpus \
--num-nodes $node_n \
--random-seed True \
