from copy import deepcopy
import sys, os

import torch
import numpy as np


from vp_suite.models import SVG
from vp_suite.utils.models import state_dicts_equal

REFERENCE_GIT_URL = "https://github.com/edenton/svg"
REPO_DIR = "SVG"


def test_impl():

    from vp_suite.models.svg_denton.svg_det import SVGDet as TheirModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = 128
    channels = 3
    rnn_dim = 256
    rnn_layers = 2
    b_size = 2

    # set up original model
    print("setting up their model")
    their_model = TheirModel(in_dim, channels, rnn_dim, rnn_layers, b_size).to(device)

    # set up our Implementation
    print("setting up our implementation")
    our_model = SVG(in_dim=in_dim, in_channels=channels, nf=64, hidden_dim=rnn_dim,
                    num_layers=rnn_layers, learned_prior=False, device=device, encoder_arch="DCGAN",
                    img_shape=(3, 64, 64), action_size=None, tensor_value_range=(0, 1)).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    assert state_dicts_equal(their_model, our_model), "State dicts not equal!"
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    assert state_dicts_equal(their_model, our_model, check_values=True), "State dicts not equal!"

    # set up input
    print("setting up input")
    their_x = torch.rand(2, 23, 3, 64, 64, device=device)
    our_x = their_x.clone()

    # infer: their model
    print("infer: theirs")
    their_x = their_x.transpose(0, 1)
    their_out = their_model(their_x, n_pred=17)

    # infer: our model
    print("infer: ours")
    our_out, _ = our_model(our_x, pred_frames=17, teacher_force=True)

    # checks
    print("check results")
    for (theirs, ours) in zip(their_out, our_out):
        theirs = theirs.detach().cpu().numpy()
        ours = ours.detach().cpu().numpy()
        assert theirs.shape == ours.shape, f"Prediction shapes are not equal. " \
                                           f"Theirs: {theirs.shape}, ours: {ours.shape}"
        assert np.allclose(theirs, ours, rtol=0, atol=1e-4), "Predictions are not equal."

    print("SVG-Det implementation matches original")
    return


if __name__ == '__main__':
    test_impl()
