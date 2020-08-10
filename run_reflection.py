import os

TRAIN_DIR = 'temp_online_ckpt/'
TRAINING_DATA_PATH = 'reflection_imgs/'
TRAINING_SCENE = '00000'
GPU_ID = '0'
IMG_TYPE = 'png'
OUTPUT_DIR = 'output'
OPTIMIZATION_STEPS = 200

# run online optimization
os.system('python3 train_reflection_online.py --train_dir '+TRAIN_DIR+' --training_data_path '+TRAINING_DATA_PATH+' --training_scene '+TRAINING_SCENE+' --GPU_ID '+GPU_ID+' --max_steps '+str(OPTIMIZATION_STEPS+10))

# inference with the optimized weights
os.system('python3 test_reflection.py --test_dataset_name '+TRAINING_DATA_PATH+'/'+TRAINING_SCENE+' --img_type '+IMG_TYPE+' --ckpt_path '+TRAIN_DIR+'model.ckpt-'+str(OPTIMIZATION_STEPS)+' --output_dir '+OUTPUT_DIR)
