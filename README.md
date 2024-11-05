# SVSRD Spatial Visual and Statistical Relation Distillation for Class-Incremental Semantic Segmentation
## Pre-requisites
Python (3.7) <br> 
Pytorch (1.8.1) <br> 
## Training
```
# An example srcipt for 15-5 overlapped setting of PASCAL VOC

GPU=0,1,2,3
BS=8  # Total 32
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-5' # or ['15-1', '19-1', '10-1', '5-3']
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=100

NAME='DKD'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

## Testing
```
python eval_voc.py -d 0 -r path/to/weight.pth
```

The complete code will be released after the paper is published.
Thank you for your interest in our work。
