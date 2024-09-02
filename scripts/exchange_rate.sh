# most of these parameters are default
# python run.py \
#     --n_features 7 \
#     --data_path Exchange_Rate_Report.csv \
#     --model DLinear \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 24\
#     --itrs 3\
#     --seed 2024\
#     --batch_size 32\
#     --train_epochs 10

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
    --model $model --itrs $itrs
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
    --factor 3 --channel_independence 1 --itrs $itrs