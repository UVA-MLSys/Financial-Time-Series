#!/usr/bin/env bash
#SBATCH --job-name="exchange_rate"
#SBATCH --output=outputs/exchange_rate2.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=(DLinear PatchTST TimesNet iTransformer)
data_path=Exchange_Rate_Report.csv
n_features=7
itrs=3

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features $n_features \
    --data_path $data_path\
    --model $model --itrs $itrs --disable_progress
done

# MICN requires label_len to be equal to seq_len
python run.py \
    --n_features $n_features \
    --data_path $data_path\
    --model MICN --disable_progress --itrs $itrs --label_len 96

python run.py\
    --model TimeMixer\
    --n_features $n_features --data_path $data_path\
    --down_sampling_layers 3 --down_sampling_window 2\
    --d_model 16 --d_ff 32 --label_len 0 \
    --down_sampling_method avg --e_layers 3 \
    --factor 3 --channel_independence 1 --itrs $itrs  --disable_progress

python run_CALF.py\
    --n_features $n_features --d_model 768\
    --data_path $data_path\
    --disable_progress --itrs $itrs\
    --model_id ori

python run_OFA.py\
    --n_features $n_features \
    --data_path $data_path\
    --disable_progress --itrs $itrs --d_model 768\
    --model_id ori