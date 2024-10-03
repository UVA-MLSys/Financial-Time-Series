#!/usr/bin/env bash
#SBATCH --job-name="48_Crude_Oil"
#SBATCH --output=outputs/Horizon_48_Crude_Oil.out
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=(DLinear PatchTST TimesNet iTransformer)
data_paths=('Apple.csv' 'Crude_Oil.csv' 'Exchange_Rate_Report.csv' 'Gold.csv' 'MSFT.csv' 'Natural_Gas.csv' 'SPX500.csv')
itrs=3
pred_len=48

for data_path in ${data_paths[@]}
do 
    if [ $data_path == 'Exchange_Rate_Report.csv' ]
    then 
        n_features=7
    elif [ $data_path == 'SPX500.csv' ]
    then 
        n_features=4
    else
        n_features=5
    fi
    

    for model in ${models[@]}
    do 
    echo "Running for model:$model"

    python run.py \
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --model $model --itrs $itrs --disable_progress

    done

    # MICN requires label_len to be equal to seq_len
    python run.py \
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --model MICN --disable_progress --itrs $itrs --label_len 96

    python run.py\
        --model TimeMixer\
        --n_features $n_features --data_path $data_path\
        --down_sampling_layers 3 --down_sampling_window 2\
        --d_model 16 --d_ff 32 --label_len 0 \
        --down_sampling_method avg --e_layers 3 \
        --factor 3 --channel_independence 1 --itrs $itrs --pred_len $pred_len --disable_progress

    python run_CALF.py\
        --n_features $n_features --d_model 768\
        --data_path $data_path --disable_progress\
        --itrs $itrs --pred_len $pred_len\
        --model_id ori

    python run_OFA.py\
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --itrs $itrs --d_model 768 --disable_progress\
        --model_id ori

    python run_TimeLLM.py\
        --n_features $n_features --d_model 16\
        --data_path $data_path --pred_len $pred_len\
        --batch_size 16 --itrs $itrs --disable_progress\
        --model_id ori

done 
