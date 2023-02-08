from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sigma", type = float, default=0.05)
    parser.add_argument("--n_sample", type = int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--expl_method", choices=['IG', 'GradCAM', 'GuidedBackprop', 'InputXGrad', 'LRP', 'Occlusion', 'Saliency', 'LayerLRP', 'DeepLift', 'LayerIG', 'LayerXAct', "LayerDL"])
    parser.add_argument("--transform", choices=['spatial', 'color', 'both'])
    parser.add_argument("--sign", choices=['all', 'absolute'])
    parser.add_argument("--reduction", choices=['none', 'sum', 'mean'], default='none')

    parser.add_argument("--split", choices = ['train', 'val'])
    parser.add_argument("--upsample", action="store_true")
    '''
    orig : conf_interval is defined as [orig_expl - q_hat, orig_expl + q_hat]
    new : conf_interval is defined as [pred_expl - q_hat, pred_expl + q_hat]
    '''
    parser.add_argument("--eval_method", choices=['orig', 'new'], default='orig')


    '''
    orig : prediction expl is \phi(X_obs)
    new : prediction expl is \phi(T_c(X_obs))
    '''
    parser.add_argument("--pred_method", choices=['orig', 'new'])

    parser.add_argument("--seed", type = int, default=0)
    parser.add_argument("--date", type=str)
    parser.add_argument("--run_option", choices=['all', 'eval', 'pred'])

    parser.add_argument("--convert_device", action="store_true", default=False)

    args = parser.parse_args()

    return args