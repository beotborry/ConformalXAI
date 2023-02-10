n_sample=2000

for seed in 0 1 2
do
    for split in 'val'
    do
        for sigma in 0.05
        do
            for expl in "LayerDL"
            do
                python3 main.py --expl_method $expl --alpha 0.05 --n_sample $n_sample --device 0  \
                --eval_method "orig" --pred_method "orig" --seed $seed --transform "both" --sigma $sigma --date "${split}_seed_${seed}" \
                --reduction 'none' --sign 'all' --run_option 'pred' --reduction 'sum' --split $split
            done
        done
    done
done