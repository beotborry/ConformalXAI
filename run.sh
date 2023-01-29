for img_path in "/home/juhyeon/Imagenet/train/n02100236/n02100236_18.JPEG" "/home/juhyeon/Imagenet/train/n01443537/n01443537_605.JPEG" "/home/juhyeon/Imagenet/train/n01614925/n01614925_13.JPEG"
do
    for expl in "GradCAM" "InputXGrad" "GuidedBackprop" "LRP" "IG"
    do
        for alpha in 0.05 0.1 0.2
        do
            python3 main.py --expl_method $expl --alpha $alpha --n_sample 1000 --device 1 --img_path $img_path
        done
    done
done