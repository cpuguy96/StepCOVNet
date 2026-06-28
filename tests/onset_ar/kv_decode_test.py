import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, inference, kv_decode, models, targets


def _tiny_experiment_config() -> config.ArExperimentConfig:
    return config.ArExperimentConfig(
        dataset=config.ArDatasetConfig(max_audio_seconds=1.0, hop_sec=0.01),
        model=config.ArModelConfig(
            patch_frames=4,
            d_model=32,
            n_enc_layers=1,
            n_dec_layers=2,
            num_heads=2,
            max_decode_steps=16,
            delta_max_dense=8,
            n_log_buckets=4,
            n_first_abs_bins=8,
            dropout_rate=0.0,
        ),
        run=config.ArRunConfig(epochs=1, model_output_dir=""),
    )


class KvDecodeTest(unittest.TestCase):
    def test_kv_decode_step0_logits_are_close_to_prefix(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()
        patch_dim = experiment_config.patch_input_dim()

        rng = np.random.default_rng(0)
        mert_patches = rng.standard_normal((1, max_patches, patch_dim)).astype(
            np.float32
        )
        patch_mask = np.zeros((1, max_patches), dtype=np.float32)
        patch_mask[0, :6] = 1.0

        encoder, decoder = models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        memory = encoder(
            {"mert_patches": mert_patches, "patch_mask": patch_mask},
            training=False,
        ).numpy()
        dec_in = np.zeros((1, max_dec), dtype=np.int32)
        dec_mask = np.zeros((1, max_dec), dtype=np.float32)
        dec_in[0, 0] = targets.BOS_ID
        dec_mask[0, 0] = 1.0
        prefix_outputs = decoder(
            {
                "encoder_memory": memory,
                "patch_mask": patch_mask,
                "decoder_input_ids": dec_in,
                "decoder_mask": dec_mask,
            },
            training=False,
        )

        kv_decoder = kv_decode.ArOnsetKvDecoder.from_model(model, experiment_config)
        patch_mask_tf = tf.constant(patch_mask, dtype=tf.float32)
        kv_decoder.precompute_cross_attention_kv(tf.constant(memory, dtype=tf.float32))
        kv_outputs, _ = kv_decoder.decode_step(
            tf.constant([[targets.BOS_ID]], dtype=tf.int32),
            0,
            patch_mask=patch_mask_tf,
            self_kv_cache=kv_decoder.initial_self_kv_cache(),
            cross_attention_mask=kv_decoder.build_cross_attention_mask(patch_mask_tf),
        )

        np.testing.assert_allclose(
            np.asarray(prefix_outputs["token_logits"][0, 0]),
            np.asarray(kv_outputs["token_logits"][0, 0]),
            rtol=1e-4,
            atol=2.0,
        )

    def test_kv_decode_matches_prefix_decode(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()
        patch_dim = experiment_config.patch_input_dim()

        rng = np.random.default_rng(0)
        mert_patches = rng.standard_normal((1, max_patches, patch_dim)).astype(
            np.float32
        )
        patch_mask = np.zeros((1, max_patches), dtype=np.float32)
        patch_mask[0, :6] = 1.0

        decode_kwargs = {
            "max_decoder_len": max_dec,
            "patch_frames": experiment_config.model.patch_frames,
            "hop_sec": experiment_config.dataset.hop_sec,
            "experiment_config": experiment_config,
            "bos_id": targets.BOS_ID,
            "eos_id": targets.EOS_ID,
        }

        prefix_stats = inference.decode_autoregressive_with_stats_numpy(
            model,
            mert_patches,
            patch_mask,
            use_kv_cache=False,
            **decode_kwargs,
        )
        kv_stats = inference.decode_autoregressive_with_stats_numpy(
            model,
            mert_patches,
            patch_mask,
            use_kv_cache=True,
            **decode_kwargs,
        )

        self.assertEqual(prefix_stats.n_forward_steps, kv_stats.n_forward_steps)
        self.assertEqual(prefix_stats.n_onset_tokens, kv_stats.n_onset_tokens)
        self.assertEqual(prefix_stats.stopped_on_eos, kv_stats.stopped_on_eos)
        self.assertGreater(kv_stats.n_onset_tokens, 0)

    def test_cross_kv_precompute_matches_cross_attn(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        kv_decoder = kv_decode.ArOnsetKvDecoder.from_model(model, experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        memory = tf.random.normal((1, max_patches, experiment_config.model.d_model))
        patch_mask = tf.concat(
            [
                tf.ones((1, 4), dtype=tf.float32),
                tf.zeros((1, max_patches - 4), dtype=tf.float32),
            ],
            axis=1,
        )
        x = tf.random.normal((1, 1, experiment_config.model.d_model))
        cross_mask = kv_decoder.build_cross_attention_mask(patch_mask)
        kv_decoder.precompute_cross_attention_kv(memory)
        for layer_idx, bundle in enumerate(kv_decoder.layer_bundles):
            direct = bundle.cross_attn(
                query=x,
                value=memory,
                key=memory,
                attention_mask=cross_mask,
                training=False,
            )
            cross_q = bundle.cross_attn.query_dense(x)
            cached = kv_decode._mha_attention_output(
                bundle.cross_attn,
                cross_q,
                kv_decoder._cross_keys[layer_idx],
                kv_decoder._cross_values[layer_idx],
                attention_mask=cross_mask,
            )
            np.testing.assert_allclose(
                np.asarray(direct),
                np.asarray(cached),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_kv_decode_step_output_shapes(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        kv_decoder = kv_decode.ArOnsetKvDecoder.from_model(model, experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        patch_dim = experiment_config.patch_input_dim()
        vocab_size = experiment_config.build_vocab().vocab_size
        mert_patches = tf.zeros((1, max_patches, patch_dim), dtype=tf.float32)
        patch_mask = tf.concat(
            [
                tf.ones((1, 4), dtype=tf.float32),
                tf.zeros((1, max_patches - 4), dtype=tf.float32),
            ],
            axis=1,
        )
        memory = models._encode_patches(
            mert_patches,
            patch_mask,
            max_patches=max_patches,
            d_model=experiment_config.model.d_model,
            num_heads=experiment_config.model.num_heads,
            n_enc_layers=experiment_config.model.n_enc_layers,
            dropout_rate=0.0,
        )
        cross_mask = kv_decoder.build_cross_attention_mask(patch_mask)
        kv_decoder.precompute_cross_attention_kv(memory)
        outputs, _ = kv_decoder.decode_step(
            tf.constant([[targets.BOS_ID]], dtype=tf.int32),
            0,
            patch_mask=patch_mask,
            self_kv_cache=kv_decoder.initial_self_kv_cache(),
            cross_attention_mask=cross_mask,
        )
        self.assertEqual(outputs["token_logits"].shape, (1, 1, vocab_size))
        self.assertEqual(outputs["pointer_logits"].shape, (1, 1, max_patches))
        self.assertEqual(outputs["residual_sec"].shape, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
