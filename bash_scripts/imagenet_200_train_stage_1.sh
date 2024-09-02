export OMP_NUM_THREADS=4

HOME=/data/workspace/GCD/PromptCAL
DATASET=imagenet_200
EXP_NAME='log-'${DATASET}'-surveillance'
EXP_ID='stage1-gpu-test_1'


echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log


SAVE_DIR=${HOME}/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=${HOME}/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

# nohup \
python -m methods.contrastive_training.contrastive_training_1 \
            --device 'cuda:1' \
            --devices '1,2' \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --lr 0.1 \
            --epochs 200 \
            --dataset_name ${DATASET} \
            --prop_train_labels 0.5 \
            --use_val 'False' \
            --val_split 0.10 \
            --batch_size 64 \
            --fast_kmeans_batch_size 10000\
            --kmeans_interval -1 \
            --eval_interval 10 \
            --use_fast_kmeans 'True' \
            --inkd_T 5 \
            --w_inkd_loss 0.5 \
            --w_inkd_loss_min 0.01 \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --num_prompts 5 \
            --early_stop -2 \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &