"""ResNet-50 inference plumbing for the face-inversion encoding GLM.

Sibling of ``cornet_utils.py``: the preprocessing, video decoding and Haar
face-crop helpers are *entirely model-agnostic*, so they are imported (re-used)
straight from ``cornet_utils`` rather than duplicated here. The only thing this
module changes is the model itself — torchvision's ImageNet-pretrained
**ResNet-50** — and the set of layers whose activations are captured.

Layer naming. We keep ResNet's own stage names ``layer1..layer4`` (rather than
relabelling them as cortical areas) to stay honest about what the model is, but
each residual stage maps onto a ventral-stream stage the CORnet notebook uses,
so the two runs can be read side by side:

    layer1  <->  V1   (early, high spatial resolution, low-level features)
    layer2  <->  V2
    layer3  <->  V4
    layer4  <->  IT   (late, low spatial resolution, object/face-level features)

Unlike CORnet-S (He/DiCarlo lab), ResNet (He et al., 2016) was not designed as
an explicit model of the ventral stream; it is a standard object-recognition
CNN we map onto those areas purely for a like-for-like comparison.

As in ``cornet_utils`` the heavy imports (``torch``, ``torchvision``) are
deferred into the function that needs them, so importing this module stays cheap
and needs no GPU.
"""

from __future__ import annotations

# The preprocessing / video / face-crop plumbing is reused verbatim: it is the
# same ImageNet 224 transform, OpenCV video decode and Haar face crop the CORnet
# path uses, and ``frame_to_features`` is model-agnostic (it just runs whatever
# model + hooks it is handed). Re-exported so the notebook's ``ru.*`` calls work.
from cornet_utils import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    USE_STABILISED_VIDEO,
    _find_data_root,
    _patch_torch_load_cpu,
    _get_transform,
    frame_to_features,
    video_path,
    open_video,
    grab_frame,
    detect_face_bbox,
    crop_to_face,
)

# ResNet-50's four residual stages, early -> late. See the module docstring for
# the layer1..layer4 <-> V1/V2/V4/IT mapping used to read this alongside CORnet.
LAYERS = ["layer1", "layer2", "layer3", "layer4"]

# Documented ventral-stream mapping (for annotations / cross-reading the plots).
LAYER_TO_AREA = {"layer1": "V1", "layer2": "V2", "layer3": "V4", "layer4": "IT"}


def load_model_and_hooks(layers: list[str] = LAYERS):
    """Load ImageNet-pretrained ResNet-50 on CPU with forward hooks on the residual stages.

    Returns ``(model, activations, handles)`` with the exact same contract as
    ``cornet_utils.load_model_and_hooks``: ``activations`` is a dict the hooks
    write each stage's raw output tensor into on every forward pass, and
    ``handles`` are the hook handles (call ``.remove()`` to detach). Unlike
    CORnet, torchvision's ResNet is not wrapped in ``nn.DataParallel``, so the
    stages are top-level submodules reached directly by name (no ``.module``).
    """
    import torchvision

    _patch_torch_load_cpu()
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    activations: dict[str, "object"] = {}

    def _get_activation(name):
        def hook(_module, _inp, output):
            activations[name] = output.detach()
        return hook

    handles = [
        getattr(model, layer).register_forward_hook(_get_activation(layer))
        for layer in layers
    ]
    return model, activations, handles
