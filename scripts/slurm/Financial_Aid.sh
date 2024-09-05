#!/usr/bin/env bash
#SBATCH --job-name="Financial Aid"
#SBATCH --output=outputs/Financial_Aid.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#---SBATCH --mem=16GB
#SBATCH --mem-per-gpu=11GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=(DLinear PatchTST TimesNet iTransformer)
data_path=Financial_Aid.csv
n_features=4
features=MS
seq_len=5
label_len=3
pred_len=1
itrs=3
target=OFFER_BALANCE
model=DLinear
python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path\
    --model $model --itrs $itrs\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target --disable_progress