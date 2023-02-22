# python3 make_results.py --seed 0 --expl_method "GradCAM" --orig_input_method "center_crop_224" --dataset "center_crop_224" --device 1


python3 aopc.py --seed 0 --device 0 --num_data 1000 --expl_method "GradCAM" --mode "insertion" --tester "OrigAOPC" --perturb_num 125 --perturb_iter 8


python3 make_results.py --seed 0 --expl_method "LayerXAct" --orig_input_method "center_crop_224" --dataset "center_crop_224" --device 1
python3 make_results.py --seed 0 --expl_method "LayerDL" --orig_input_method "center_crop_224" --dataset "center_crop_224" --device 1
python3 make_results.py --seed 0 --expl_method "LayerIG" --orig_input_method "center_crop_224" --dataset "center_crop_224" --device 1

python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerXAct" --mode "insertion" --tester "ConfAOPC" --perturb_num 125 --perturb_iter 8
python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerDL" --mode "insertion" --tester "ConfAOPC" --perturb_num 125 --perturb_iter 8
python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerIG" --mode "insertion" --tester "ConfAOPC" --perturb_num 125 --perturb_iter 8