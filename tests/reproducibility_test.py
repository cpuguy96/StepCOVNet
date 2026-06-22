import numpy as np

from stepcovnet import models, reproducibility


def test_apply_training_seed_stabilizes_model_init() -> None:
    reproducibility.apply_training_seed(123)
    model_a = models.build_unet_wavenet_model(
        depth=1,
        initial_filters=8,
        input_features=16,
    )
    weights_a = model_a.get_weights()[0]

    reproducibility.apply_training_seed(123)
    model_b = models.build_unet_wavenet_model(
        depth=1,
        initial_filters=8,
        input_features=16,
    )
    weights_b = model_b.get_weights()[0]

    np.testing.assert_array_equal(weights_a, weights_b)


def test_different_seeds_change_model_init() -> None:
    reproducibility.apply_training_seed(1)
    model_a = models.build_unet_wavenet_model(
        depth=1,
        initial_filters=8,
        input_features=16,
    )
    weights_a = model_a.get_weights()[0]

    reproducibility.apply_training_seed(2)
    model_b = models.build_unet_wavenet_model(
        depth=1,
        initial_filters=8,
        input_features=16,
    )
    weights_b = model_b.get_weights()[0]

    assert not np.allclose(weights_a, weights_b)
