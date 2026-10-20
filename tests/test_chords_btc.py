from __future__ import annotations

import io

import numpy as np
import pytest

from songscribe import chords
from songscribe.chords import BTC_QUALITIES, btc_label, normalise_label, recognise


class TestLabels:
    def test_every_class_has_a_distinct_label(self):
        labels = [btc_label(i) for i in range(170)]
        assert len(set(labels)) == 170

    def test_roots_and_qualities_run_in_checkpoint_order(self):
        assert btc_label(0) == "C:min"
        assert btc_label(1) == "C:maj"
        assert btc_label(14) == "C#:min"
        assert btc_label(9) == "C:7"
        assert btc_label(11 * 14 + 8) == "B:maj7"

    def test_the_last_two_classes_are_unknown_and_no_chord(self):
        assert btc_label(168) == "X"
        assert btc_label(169) == "N"

    def test_every_quality_reaches_the_chart_vocabulary(self):
        # normalise_label passes an unknown quality through; a quality left as Harte
        # text would print as "C:something" on a chart.
        for quality in BTC_QUALITIES:
            assert ":" not in normalise_label(f"C:{quality}"), quality

    def test_sevenths_survive_normalisation(self):
        assert normalise_label(btc_label(1 * 14 + 6)) == "C#m7"
        assert normalise_label(btc_label(7 * 14 + 9)) == "G7"
        assert normalise_label(btc_label(0 * 14 + 8)) == "Cmaj7"


class TestRegistration:
    def test_runs_at_the_rate_the_checkpoint_was_trained_on(self):
        assert chords.sample_rate_for("btc") == chords.BTC_SAMPLE_RATE

    def test_a_wrong_sample_rate_is_refused_before_any_model_loads(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="22050"):
            recognise(np.zeros(44_100, dtype=np.float32), 44_100, backend="btc")

    def test_an_unknown_device_is_refused(self):
        with pytest.raises(ValueError, match="device"):
            recognise(np.zeros(22_050, dtype=np.float32), 22_050, backend="btc", device="tpu")


@pytest.fixture
def btc():
    pytest.importorskip("torch")
    from songscribe import btc

    return btc


class TestNetwork:
    def test_gives_one_logit_vector_per_frame(self, btc):
        import torch

        model = btc.BTC().eval()
        with torch.no_grad():
            out = model(torch.randn(2, btc.TIMESTEPS, btc.N_BINS))
        assert out.shape == (2, btc.TIMESTEPS, btc.NUM_CHORDS)

    def test_backward_attention_looks_only_ahead_and_forward_only_back(self, btc):
        model = btc.BTC()
        layer = model.self_attn_layers.self_attn_layers[0]
        forward = layer.attn_block.multi_head_attention.mask
        backward = layer.backward_attn_block.multi_head_attention.mask
        assert forward[0, 1] == -np.inf and forward[1, 0] == 0
        assert backward[1, 0] == -np.inf and backward[0, 1] == 0

    def test_features_have_one_frame_per_hop(self, btc):
        assert chords.BTC_SAMPLE_RATE == btc.SAMPLE_RATE
        audio = np.zeros(btc.SAMPLE_RATE * 25, dtype=np.float32)
        feature = btc.features(audio)
        assert feature.shape[1] == btc.N_BINS
        # Two full 10 s chunks of 108 frames, and the 5 s remainder.
        assert feature.shape[0] == 108 * 2 + 1 + 5 * btc.SAMPLE_RATE // btc.HOP_LENGTH

    def test_predict_returns_a_class_for_every_frame_not_the_padding(self, btc):
        model = btc.BTC().eval()
        classes = btc.predict(model, 0.0, 1.0, np.zeros((150, btc.N_BINS)), "cpu")
        assert classes.shape == (150,)


class TestCheckpointDownload:
    def test_a_download_with_the_wrong_checksum_is_refused_and_not_cached(
        self, btc, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(btc.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"junk"))
        with pytest.raises(RuntimeError, match="SHA-256"):
            btc.checkpoint_path(tmp_path)
        assert not list(tmp_path.rglob("*.pt"))

    def test_a_cached_checkpoint_is_not_downloaded_again(self, btc, monkeypatch, tmp_path):
        cached = tmp_path / "songscribe" / f"btc_large_voca_{btc.UPSTREAM_COMMIT[:8]}.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"x")

        def no_network(*args, **kwargs):
            raise AssertionError("downloaded")

        monkeypatch.setattr(btc.urllib.request, "urlopen", no_network)
        assert btc.checkpoint_path(tmp_path) == cached

    def test_a_missing_checkpoint_path_names_the_file(self, btc, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"nope\.pt"):
            btc.load(tmp_path / "nope.pt", "cpu")
