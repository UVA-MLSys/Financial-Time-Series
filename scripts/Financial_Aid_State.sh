# models=(DLinear PatchTST TimesNet iTransformer)
models=(DLinear)
data_path=Financial_Aid_State.csv
n_features=2
features=M
seq_len=5
label_len=3
pred_len=1
itrs=3
top_k=2
patch_len=4
stride=2

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id GROUP_ID\
    --model $model --itrs $itrs\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --top_k $top_k --freq a
done

# MICN requires label_len to be equal to seq_len
python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id GROUP_ID\
    --model MICN --disable_progress --itrs $itrs\
    --seq_len $seq_len --label_len $seq_len --pred_len $pred_len\
    --freq a

python run.py\
    --model TimeMixer\
    --n_features $n_features --data_path $data_path --group_id GROUP_ID\
    --down_sampling_layers 3 --down_sampling_window 2\
    --d_model 16 --d_ff 32\
    --seq_len $seq_len --label_len 0 --pred_len $pred_len\
    --down_sampling_method avg --e_layers 3 --freq a\
    --factor 3 --channel_independence 1 --itrs $itrs  --features $features

python run_CALF.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id GROUP_ID\
    --itrs $itrs\
    --model_id ori --freq a --d_model 768\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len

python run_OFA.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id GROUP_ID\
    --itrs $itrs --d_model 768\
    --model_id ori --patch_size $patch_len --stride $stride\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len

python run_TimeLLM.py\
    --n_features $n_features --d_model 16\
    --data_path $data_path --group_id GROUP_ID --freq a \
    --batch_size 16 --itrs $itrs --disable_progress\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --model_id ori --top_k $top_k\
    --patch_size $patch_len --stride $stride