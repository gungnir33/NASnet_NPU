alias python="/usr/bin/python3.7.5"
source "/home/NASnet_mobile/scripts/env.sh"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
nohup python3.7.5 -u main_npu_8p.py \
	"/home/data/imagenet/" \
        -a=nasnetamobile \
        --addr=$(hostname -I |awk '{print $1}') \
        --dist-url 'tcp://127.0.0.1:26999' \
        --lr=1.5 \
        --wd=1.0e-04 \
        --momentum=0.9 \
        --print-freq=10 \
        --epochs=100 \
        --workers=$(nproc) \
        --seed=49 \
        --amp \
        --world-size=1 \
        --dist-backend='hccl' \
        --loss-scale=128.0 \
        --opt-level='O2' \
        --device='npu' \
        --multiprocessing-distributed \
        --rank=0 \
        --label-smoothing=0.1 \
        --warm_up_epochs=3 \
        --batch-size=3072 > ./log/nasnet_a_mobile_npu_8p.log 2>&1 &