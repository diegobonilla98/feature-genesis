import torch

from feature_genesis.sae.metrics import (
    maximum_positive_decoder_cosine,
    mutual_coherence,
)


def test_positive_decoder_cosine_does_not_treat_antipodal_features_as_duplicates():
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert mutual_coherence(directions) == 1.0
    assert maximum_positive_decoder_cosine(directions) == 0.0


def test_positive_decoder_cosine_detects_parallel_features():
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert maximum_positive_decoder_cosine(directions) == 1.0
