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

models=(DLinear MICN PatchTST SegRNN TimeMixer TimesNet iTransformer)

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features 7 \
    --data_path Exchange_Rate_Report.csv\
    --model $model --itrs 3
done