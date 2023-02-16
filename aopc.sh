python3 aopc.py --seed 0 --num_data 1000 --expl_method "GradCAM" --mode "insertion" --tester "ConfAOPC" --transform "spatial"
python3 aopc.py --seed 0 --num_data 1000 --expl_method "GradCAM" --mode "insertion" --tester "ConfAOPC" --transform "spatial" "noise" "color"

python3 aopc.py --seed 0 --num_data 1000 --expl_method "LayerXAct" --mode "insertion" --tester "ConfAOPC" --transform "spatial"
python3 aopc.py --seed 0 --num_data 1000 --expl_method "LayerXAct" --mode "insertion" --tester "ConfAOPC" --transform "spatial" "noise" "color"

python3 aopc.py --seed 0 --num_data 1000 --expl_method "LayerDL" --mode "insertion" --tester "ConfAOPC" --transform "spatial"
python3 aopc.py --seed 0 --num_data 1000 --expl_method "LayerDL" --mode "insertion" --tester "ConfAOPC" --transform "spatial" "noise" "color"
