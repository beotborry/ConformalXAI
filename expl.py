import functools
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients, InputXGradient, GuidedBackprop, LRP
from captum.attr import visualization as viz


class ExplFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_explainer(model, expl_method, layer = None):
        if layer is None:
            layer = model.layer4 # last conv layer

        assert expl_method is not None
        
        if expl_method == "GradCAM":
            return functools.partial(get_grad_cam, model, layer)
        elif expl_method == "IG":
            return functools.partial(get_ig, model)
        elif expl_method == "InputXGrad":
            return functools.partial(get_input_x_grad, model)
        elif expl_method == "GuidedBackprop":
            return functools.partial(get_gbp, model)
        elif expl_method == "LRP":
            return functools.partial(get_lrp, model)

def get_grad_cam(model, layer, img):
    target = model(img).argmax()

    gc = LayerGradCam(model, layer)
    attr = gc.attribute(img, target = target)
    return LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear')

def get_ig(model, img):
    target = model(img).argmax()

    ig = IntegratedGradients(model)
    return ig.attribute(img, target = target)

def get_input_x_grad(model, img):
    target = model(img).argmax()

    input_x_grad = InputXGradient(model)
    return input_x_grad.attribute(img, target = target)

def get_gbp(model, img):
    target = model(img).argmax()

    gbp = GuidedBackprop(model)
    return gbp.attribute(img, target = target)

def get_lrp(model, img):
    target = model(img).argmax()

    lrp = LRP(model)
    return lrp.attribute(img, target = target)