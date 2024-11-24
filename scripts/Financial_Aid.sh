models=(DLinear PatchTST TimesNet iTransformer)
data_path=Financial_Aid.csv
n_features=4
features=MS
seq_len=5
label_len=3
pred_len=1
itrs=3
top_k=2
patch_len=4
stride=2
target=OFFER_BALANCE
freq=a
group_id=GROUP_ID

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id $group_id\
    --model $model --itrs $itrs\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target --top_k $top_k --freq $freq
done

# MICN requires label_len to be equal to seq_len
python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id $group_id\
    --model MICN --disable_progress --itrs $itrs\
    --seq_len $seq_len --label_len $seq_len --pred_len $pred_len\
    --target $target --freq $freq

python run.py\
    --model TimeMixer\
    --n_features $n_features --data_path $data_path --group_id $group_id\
    --down_sampling_layers 3 --down_sampling_window 2\
    --d_model 16 --d_ff 32\
    --seq_len $seq_len --label_len 0 --pred_len $pred_len\
    --down_sampling_method avg --e_layers 3 --freq $freq\
    --factor 3 --channel_independence 1 --itrs $itrs  --features $features\
    --target $target

python run_CALF.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id $group_id\
    --itrs $itrs\
    --model_id ori --freq $freq --d_model 768\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target

python run_OFA.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id $group_id\
    --itrs $itrs --d_model 768\
    --model_id ori --patch_size $patch_len --stride $stride\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target

python run_TimeLLM.py\
    --n_features $n_features --d_model 16\
    --data_path $data_path --group_id $group_id --freq $freq \
    --batch_size 16 --itrs $itrs --disable_progress\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --model_id ori --target $target --top_k $top_k\
    --patch_size $patch_len --stride $stride