data_path=Financial_Aid_State.csv
n_features=1
features=S
target=need_amt
seq_len=10
label_len=5
pred_len=1
itrs=3
top_k=2
patch_len=4
stride=2
freq=a
group_id=GROUP_ID
percents=(25 50 75)
models=(DLinear PatchTST TimesNet iTransformer)

for percent in ${percents[@]}
do 
    for model in ${models[@]}
    do 

    python run.py \
        --n_features $n_features --features $features\
        --data_path $data_path --group_id $group_id\
        --model $model --itrs $itrs\
        --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
        --top_k $top_k --freq $freq --target $target\
        --percent $percent --channel_independence 0
    done

    python run.py\
        --model TimeMixer\
        --n_features $n_features --data_path $data_path --group_id $group_id\
        --down_sampling_layers 3 --down_sampling_window 1\
        --d_model 16 --d_ff 32\
        --seq_len $seq_len --label_len 0 --pred_len $pred_len\
        --down_sampling_method avg --e_layers 3 --freq $freq\
        --factor 3 --channel_independence 0 --itrs $itrs \
        --features $features --target $target --percent $percent

    python run_CALF.py\
        --n_features $n_features --features $features \
        --data_path $data_path --group_id $group_id\
        --itrs $itrs\
        --model_id ori --freq $freq --d_model 768\
        --seq_len $seq_len --label_len $label_len\
        --pred_len $pred_len --target $target --percent $percent

    python run_OFA.py\
        --n_features $n_features --features $features \
        --data_path $data_path --group_id $group_id\
        --itrs $itrs --d_model 768\
        --model_id ori --patch_size $patch_len --stride $stride\
        --seq_len $seq_len --label_len $label_len --freq $freq\
        --pred_len $pred_len --target $target --percent $percent

    python run_TimeLLM.py\
        --n_features $n_features --d_model 16\
        --data_path $data_path --group_id $group_id --freq $freq \
        --batch_size 16 --itrs $itrs --disable_progress\
        --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
        --model_id ori --top_k $top_k\
        --patch_len $patch_len --stride $stride --target $target --percent $percent

done