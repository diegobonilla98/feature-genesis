from types import SimpleNamespace

import torch

from feature_genesis.circuits.sparse import (
    contribution_scores,
    encode_spatial,
    max_feature_values,
    modify_sparse_features,
)
from feature_genesis.sae.topk import TopKSAE


def identity_sae():
    sae = TopKSAE(2, 2, 2, 0.0, 2, 10)
    with torch.no_grad():
        sae.encoder.copy_(torch.eye(2))
        sae.decoder.copy_(torch.eye(2))
        sae.encoder_bias.zero_()
        sae.decoder_bias.zero_()
        sae.input_scale.fill_(1)
    return sae


def test_sparse_spatial_encoding_and_interventions():
    sae = identity_sae()
    activation = torch.tensor([[[[2.0]], [[3.0]]]])
    codes, shape = encode_spatial(sae, activation)
    assert shape == (1, 2, 1, 1)
    assert torch.equal(max_feature_values(codes, 0), torch.tensor([2.0]))
    ablated = modify_sparse_features(activation, sae, [0], "ablate_selected")
    kept = modify_sparse_features(activation, sae, [0], "keep_selected")
    removed = modify_sparse_features(activation, sae, [0], "remove_all")
    assert torch.allclose(ablated, torch.tensor([[[[0.0]], [[3.0]]]]))
    assert torch.allclose(kept, torch.tensor([[[[2.0]], [[0.0]]]]))
    assert torch.allclose(removed, torch.zeros_like(activation))


def test_contribution_scores_project_gradient_into_sparse_basis():
    sae = identity_sae()
    activation = torch.tensor([[[[2.0]], [[3.0]]]])
    gradient = torch.tensor([[[[5.0]], [[7.0]]]])
    scores = contribution_scores(activation, gradient, sae)
    assert torch.allclose(scores, torch.tensor([[10.0, 21.0]]))
