n_sample=2000

for seed in 0 1 2 3 4
do
    for split in 'val'
    do
        for sigma in 0.05
        do
            for expl in "GradCAM"
            do
                CUDA_VISIBLE_DEVICES=0,1 python3 main.py --expl_method $expl --alpha 0.05 --n_sample $n_sample --device 0 1 \
                --eval_method "orig" --pred_method "orig" --seed $seed --transform "both" --sigma $sigma --date "${split}_seed_${seed}" \
                --reduction 'none' --sign 'all' --run_option 'pred' --reduction 'sum' --split $split --dataset "center_crop_224" --orig_input_method "center_crop_224" \
                --batch_size 100
            done
        done
    done
done