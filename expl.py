import functools
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients, InputXGradient, GuidedBackprop, LRP, Occlusion, Saliency, DeepLift, LayerIntegratedGradients, LayerGradientXActivation, LayerConductance, LayerDeepLift, LayerActivation
from captum.attr import LayerLRP, NoiseTunnel
from captum.attr import visualization as viz
from torch.nn import Softmax


class ExplFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_explainer(model, expl_method, layer = None, upsample=False):
        if layer is None:
            layer = model.layer4 # last conv layer

        assert expl_method is not None
        
        if expl_method == "GradCAM":
            return functools.partial(get_grad_cam, model, layer, upsample)
        elif expl_method == "IG":
            return functools.partial(get_ig, model)
        elif expl_method == "InputXGrad":
            return functools.partial(get_input_x_grad, model)
        elif expl_method == "GuidedBackprop":
            return functools.partial(get_gbp, model)
        elif expl_method == "LRP":
            return functools.partial(get_lrp, model)
        elif expl_method == "Occlusion":
            return functools.partial(get_occlusion, model)
        elif expl_method == "Saliency":
            return functools.partial(get_saliency, model)
        elif expl_method == "LayerLRP":
            return functools.partial(get_layer_lrp, model, layer)
        elif expl_method == "DeepLift":
            return functools.partial(get_deeplift, model)
        elif expl_method == "LayerIG":
            return functools.partial(get_layerIG, model, layer, upsample)
        elif expl_method == "LayerXAct":
            return functools.partial(get_layerXAct, model, layer, upsample)
        elif expl_method == "LayerConductance":
            return functools.partial(get_layerConductance, model, layer)
        elif expl_method == "LayerDL":
            return functools.partial(get_layerDL, model, layer, upsample)
        else:
            print("Check Expl Method Name!")

def get_grad_cam(model, layer, upsample, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    gc = LayerGradCam(model, layer)
    attr = gc.attribute(img, target = target)
    if upsample:
        return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear'), target
    else: return attr, probs, target

def get_ig(model, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    ig = IntegratedGradients(model)
    return ig.attribute(img, target = target), target

def get_input_x_grad(model, img, orig_target=None):
    target = model(img).argmax()

    if orig_target != "init" and target != orig_target:
        return None, target

    input_x_grad = InputXGradient(model)
    return input_x_grad.attribute(img, target = target), target

def get_gbp(model, img, orig_target=None):
    target = model(img).argmax()

    if orig_target != "init" and target != orig_target:
        return None, target

    gbp = GuidedBackprop(model)
    return gbp.attribute(img, target = target), target

def get_lrp(model, img, orig_target=None):
    target = model(img).argmax()

    if orig_target != "init" and target != orig_target:
        return None, target

    lrp = LRP(model)
    return lrp.attribute(img, target = target), target

def get_occlusion(model, img, orig_target=None):
    target = model(img).argmax()
    
    if orig_target != "init" and target != orig_target:
        return None, target

    ablator = Occlusion(model)
    return ablator.attribute(img, target=target, sliding_window_shapes=(3,15,15), strides = (3, 8, 8)), target

def get_saliency(model, img, orig_target=None):
    target = model(img).argmax()
    
    if orig_target != "init" and target != orig_target:
        return None, target

    saliency = Saliency(model)

    return saliency.attribute(img, target=target), target

def get_layer_lrp(model, layer, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)


    lrp = LayerLRP(model, layer)
    attr = lrp.attribute(img, target = target)
    return attr, probs, target

def get_deeplift(model, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    dl = DeepLift(model)
    return dl.attribute(img, target=target), probs, target

def get_layerIG(model, layer, upsample, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    lig = LayerIntegratedGradients(model, layer)
    attr = lig.attribute(img, target = target)
    if upsample:
        return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear'), target
    else:
        return attr, probs, target

def get_layerXAct(model, layer, upsample, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    layer_ga = LayerGradientXActivation(model, layer)
    attr = layer_ga.attribute(img, target=target)
    if upsample:
        return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear'), target
    else:
        return attr, probs, target


def get_layerConductance(model, layer, img, orig_target=None):
    target = model(img).argmax()

    if orig_target != "init" and target != orig_target:
        return None, target
    layer_cond = LayerConductance(model, layer)

    return layer_cond.attribute(img, target=target), target

def get_layerDL(model, layer, upsample, img, orig_target=None):
    probs = Softmax(dim=1)(model(img))
    target = probs.argmax(dim = 1)

    dl = LayerDeepLift(model, layer)
    attr = dl.attribute(img, target=target)
    if upsample:
        return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear'), target
    else:
        return attr, probs, target

def get_layerIG_SG(model, layer, upsample, img, orig_target=None):
    target = model(img).argmax()

    if orig_target != "init" and target != orig_target:
        return None, target
    
    lig = LayerIntegratedGradients(model, layer)
    nt = NoiseTunnel(lig)


    attr = nt.attribute(img, target = target, nt_type='smoothgrad', nt_samples=50)
    if upsample:
        return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear'), target
    else:
        return attr, target

