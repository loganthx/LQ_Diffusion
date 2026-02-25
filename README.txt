# # # # # # # # # # # # # # #
# L_simple = 'MSE' (default)#
# L_hybrid = 'RESCALED_MSE' #
# L_vlb = 'KL'              #
# # # # # # # # # # # # # # # 

0.0 INSTALL IMPROVED-DIFFUSION:
pip install -e benchmark/improved-diffusion



1.1 BUILD DATA:
python dataset/lq_dataset.py

SESSION_NAME='2'
SCHEDULE_NAME='logistic'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64

1.2 TRAIN:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
TRAIN_FLAGS="--lr 2e-4 --batch_size ${BATCH_SIZE}"
python3 setup.py scripts.image_train --data_dir built_data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

1.3 SAMPLE:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
python3 setup.py scripts.image_sample --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt --num_samples ${NUM_SAMPLES} $MODEL_FLAGS $DIFFUSION_FLAGS

1.4 EXTRACT:
python3 extractor/extract.py --npz_path logs/${SCHEDULE_NAME}${SESSION_NAME}/samples_${NUM_SAMPLES}x${IMG_SIZE}x${IMG_SIZE}x3.npz \
--labels_json built_data/labels.json --out_dir logs/${SCHEDULE_NAME}${SESSION_NAME}/extracted \
--use_subdirs True --limit 500



--- FOR COMPARISON, WE TRAIN COSINE --- 

SESSION_NAME='2'
SCHEDULE_NAME='cosine'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64


2.1 TRAIN:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
TRAIN_FLAGS="--lr 2e-4 --batch_size ${BATCH_SIZE}"
python3 setup.py scripts.image_train --data_dir built_data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

2.2 SAMPLE:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
python3 setup.py scripts.image_sample --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt --num_samples ${NUM_SAMPLES} $MODEL_FLAGS $DIFFUSION_FLAGS

2.3 EXTRACT:
python3 extractor/extract.py --npz_path logs/${SCHEDULE_NAME}${SESSION_NAME}/samples_${NUM_SAMPLES}x${IMG_SIZE}x${IMG_SIZE}x3.npz \
--labels_json built_data/labels.json --out_dir logs/${SCHEDULE_NAME}${SESSION_NAME}/extracted \
--use_subdirs True --limit 500



--- FOR COMPARISON, WE TRAIN LINEAR --- 

SESSION_NAME='3'
SCHEDULE_NAME='linear'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64

3.1 TRAIN:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
TRAIN_FLAGS="--lr 2e-4 --batch_size ${BATCH_SIZE}"
python3 setup.py scripts.image_train --data_dir built_data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

3.2 SAMPLE:
export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps ${STEPS} --noise_schedule ${SCHEDULE_NAME}"
python3 setup.py scripts.image_sample --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt --num_samples ${NUM_SAMPLES} $MODEL_FLAGS $DIFFUSION_FLAGS 

3.3 EXTRACT:
python3 extractor/extract.py --npz_path logs/${SCHEDULE_NAME}${SESSION_NAME}/samples_${NUM_SAMPLES}x${IMG_SIZE}x${IMG_SIZE}x3.npz \
--labels_json built_data/labels.json --out_dir logs/${SCHEDULE_NAME}${SESSION_NAME}/extracted \
--use_subdirs True --limit 500


#########################
###     EVALUATE     ####
#########################

--- CALC NLL LINEAR --- 

SESSION_NAME='3'
SCHEDULE_NAME='linear'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64

export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}/NLL
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
python3 setup.py scripts.image_nll --data_dir built_data --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt $MODEL_FLAGS



--- CALC NLL COS --- 

SESSION_NAME='2'
SCHEDULE_NAME='cosine'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64

export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}/NLL
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
python3 setup.py scripts.image_nll --data_dir built_data --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt $MODEL_FLAGS



--- CALC NLL LOGISTIC --- 

SESSION_NAME='2'
SCHEDULE_NAME='logistic'
STEPS=5000
NUM_SAMPLES=128
IMG_SIZE=64
BATCH_SIZE=64

export OPENAI_LOG_FORMAT=stdout,log,csv
export OPENAI_LOGDIR=./logs/${SCHEDULE_NAME}${SESSION_NAME}/NLL
MODEL_FLAGS="--image_size ${IMG_SIZE} --num_channels 64 --num_res_blocks 2 --class_cond True --num_classes 7"
python3 setup.py scripts.image_nll --data_dir built_data --model_path ./logs/${SCHEDULE_NAME}${SESSION_NAME}/ema_0.9999_050000.pt $MODEL_FLAGS

