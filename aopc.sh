python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "GradCAM" --mode "insertion" --tester "ConfAOPC" --perturb_num 125
python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerXAct" --mode "insertion" --tester "ConfAOPC" --perturb_num 125
python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerDL" --mode "insertion" --tester "ConfAOPC" --perturb_num 125
python3 aopc.py --seed 0 --device 1 --num_data 1000 --expl_method "LayerIG" --mode "insertion" --tester "ConfAOPC" --perturb_num 125

