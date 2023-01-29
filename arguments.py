from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sigma", type = float, default=0.05)
    parser.add_argument("--n_sample", type = int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--expl_method", choices=['IG', 'GradCAM', 'GuidedBackprop', 'InputXGrad', 'LRP'])

    args = parser.parse_args()

    return args