"""
Manifest-based evaluation, aligned with original eval.py flow.

Usage:
python deepfake-detector/scripts/eval_manifest.py \
  --config specnn_amplitude \
  --weights specnn_amplitude \
  --manifest_csv ood_eval/manifests/ood_three_source_balanced.csv \
  --output_csv ood_eval/preds/deezer__ood_three_source_balanced.csv \
  --gpu 0 --repeat 1
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import torchaudio
from tensorflow import keras

# Keep imports/structure close to original eval.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader.global_variables import *  # noqa: F401,F403
from loader.audio import EvalAugmenter
from loader.config import ConfLoader
from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file", type=str, default="specnn_amplitude")
parser.add_argument("--weights", help="weights file, else defaults to config", type=str)
parser.add_argument("--encoder", help="eval model trained on only one encoder", type=str, default="")
parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=-1)
parser.add_argument("--steps", help="max evaluation steps", type=int, default=10_000_000)
parser.add_argument("--repeat", help="repeat each file N times", type=int, default=1)
parser.add_argument("--manifest_csv", help="CSV with filepath,target", type=str, required=True)
parser.add_argument("--output_csv", help="output predictions CSV", type=str, required=True)
parser.add_argument("--threshold", help="classification threshold", type=float, default=0.5)
parser.add_argument("--log_every", help="log every N files during prediction dump", type=int, default=100)
parser.add_argument(
    "--model_n_encoders",
    help="Encoder head size used by trained Deezer model (specnn_amplitude default=10)",
    type=int,
    default=10,
)
parser.add_argument(
    "--eval_sections",
    help="Comma-separated sections to run: all,encoder,real (default: all,encoder,real)",
    type=str,
    default="all,encoder,real",
)
args = parser.parse_args()


if not args.weights:
    args.weights = args.config
GPU = args.gpu
ENCODER = ""
if args.encoder:
    ENCODER = args.encoder

configuration = ConfLoader(CONF_PATH)
configuration.load_model(args.config)
global_conf = configuration.conf
if args.repeat > 0:
    global_conf["repeat"] = args.repeat

if GPU >= 0:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[GPU], "GPU")
    tf.config.experimental.set_memory_growth(gpus[GPU], True)
else:
    tf.config.set_visible_devices([], "GPU")


def _to_stereo_time_last(x: np.ndarray) -> np.ndarray:
    # input is (channels,time) from torchaudio
    if x.ndim == 1:
        x = np.stack([x, x], axis=0)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    if x.shape[0] > 2:
        x = x[:2, :]
    return x.T.astype(np.float32)  # (time,channels)


def _slice_or_pad_center(x: np.ndarray, target_len: int) -> np.ndarray:
    t = x.shape[0]
    if t >= target_len:
        off = (t - target_len) // 2
        return x[off : off + target_len]
    pad = target_len - t
    return np.pad(x, ((0, pad), (0, 0)), mode="constant")


class ManifestLoader:
    def __init__(self, manifest_csv: str, params: dict):
        self.params = params
        self.df = pd.read_csv(manifest_csv).copy()
        if "filepath" not in self.df.columns or "target" not in self.df.columns:
            raise ValueError("manifest CSV must contain filepath,target columns")

        # normalize optional encoder column for fake partitions in eval output
        if "encoder" not in self.df.columns:
            self.df["encoder"] = self.df.apply(
                lambda r: "manifest_fake" if int(r["target"]) == 1 else "real", axis=1
            )

        self.df["target"] = self.df["target"].astype(int)
        self.df["exists"] = self.df["filepath"].map(lambda p: Path(str(p)).exists())
        self.df = self.df[self.df["exists"]].reset_index(drop=True)

        self.fake_df = self.df[self.df["target"] == 1].copy()
        self.real_df = self.df[self.df["target"] == 0].copy()

        self.encoders = sorted([e for e in self.fake_df["encoder"].dropna().unique().tolist() if e != "real"])
        self.n_encoders = len(self.encoders)
        self.enc_to_idx = {e: i for i, e in enumerate(self.encoders)}

        self.split_mp3 = {"test": self.df["filepath"].tolist()}

    def _make_entries(self, mode="all", encoder=None):
        rows = []
        if mode == "all":
            for _, r in self.df.iterrows():
                # Deezer convention: y=1 real, y=0 fake
                y = 1 - int(r["target"])
                enc = 0
                if y == 0:  # fake sample
                    enc = self.enc_to_idx.get(str(r["encoder"]), 0)
                rows.append((str(r["filepath"]), y, enc))
        elif mode == "real":
            for _, r in self.real_df.iterrows():
                rows.append((str(r["filepath"]), 1, 0))  # same convention as original
        elif mode == "encoder":
            if encoder not in self.enc_to_idx:
                raise ValueError(f"Unknown encoder: {encoder}")
            idx = self.enc_to_idx[encoder]
            sub = self.fake_df[self.fake_df["encoder"] == encoder]
            for _, r in sub.iterrows():
                rows.append((str(r["filepath"]), 0, idx))
        else:
            raise ValueError(mode)
        return rows

    def _prepare_feat(self, path: str, augmenter: EvalAugmenter):
        wav, sr = torchaudio.load(path)
        if int(sr) != int(self.params["sr"]):
            wav = torchaudio.functional.resample(wav, int(sr), int(self.params["sr"]))
        x = _to_stereo_time_last(wav.numpy())
        x = _slice_or_pad_center(x, int(self.params["audio_slice"] * self.params["sr"]))
        x = augmenter.transform(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()
        return x

    def make_dataset(self, mode, augmenter, input_shape, batch_size, repeat=1, encoder=None):
        entries = self._make_entries(mode=mode, encoder=encoder)

        def gen():
            for path, y, enc in entries:
                for _ in range(max(1, repeat)):
                    feat = self._prepare_feat(path, augmenter)
                    feat = feat.astype(np.float32)
                    if feat.shape != input_shape:
                        # defensive reshape check
                        feat = np.reshape(feat, input_shape)
                    yield feat, (np.float32(y), np.int32(enc))

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            ),
        )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(entries)

    def dump_predictions(self, model, augmenter, threshold, output_csv, log_every=100):
        rows = []
        for i, (_, r) in enumerate(self.df.iterrows(), start=1):
            feat = self._prepare_feat(str(r["filepath"]), augmenter)
            pred = model.predict(np.expand_dims(feat, 0), verbose=0)
            if isinstance(pred, list):
                p_real = float(np.array(pred[0]).reshape(-1)[0])
            else:
                p_real = float(np.array(pred).reshape(-1)[0])
            p_fake = 1.0 - p_real
            rows.append(
                {
                    "filepath": str(r["filepath"]),
                    "target": int(r["target"]),
                    "encoder": str(r["encoder"]),
                    "pred_prob_real": p_real,
                    "pred_prob": p_fake,  # kept as fake probability for downstream tools
                    "pred_label": int(p_fake >= threshold),
                }
            )
            if i % log_every == 0:
                print(f"[PRED] {i}/{len(self.df)}")
        out_df = pd.DataFrame(rows)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        return out_df


loader = ManifestLoader(args.manifest_csv, global_conf)
augmenter = EvalAugmenter(global_conf)

# Infer input shape from first file
first_path = loader.df.iloc[0]["filepath"]
first_feat = loader._prepare_feat(str(first_path), augmenter)
input_shape = tuple(first_feat.shape)
print(f"[INFO] input_shape={input_shape} n_files={len(loader.df)} n_encoders={loader.n_encoders}")


@tf.function
def one_hot_encoder(y, depth):
    y1, y2 = y
    y1 = tf.cast(y1, tf.int32)
    idx = (1 - y1) * (y2 + 1)
    return y1, tf.one_hot(idx, depth)


it_test, n_all = loader.make_dataset(
    mode="all",
    augmenter=augmenter,
    input_shape=input_shape,
    batch_size=global_conf["batch_size"],
    repeat=global_conf["repeat"],
)
model_n_encoders = args.model_n_encoders

if not ENCODER:
    it_test = it_test.map(lambda x, y: (x, one_hot_encoder(y, model_n_encoders)))
else:
    it_test = it_test.map(lambda x, y: (x, y[0]))

if "use_raw" in global_conf:
    it_test = it_test.map(lambda x, y: (tf.expand_dims(x, -1), y))

n_encoders = model_n_encoders
if ENCODER:
    print(f"\nEncoder-specific model! `{ENCODER}`\n")
    n_encoders = None

if "use_raw" in global_conf:
    model = SimpleCNN(input_shape, global_conf, detect_encoder=n_encoders)
else:
    model = SimpleSpectrogramCNN(input_shape, global_conf, detect_encoder=n_encoders)

model.m.load_weights(os.path.join(WEIGHTS_PATH, args.weights + args.encoder))
if not ENCODER:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss_weights={"deepfake": 1.0, "encoder": 0.2},
        metrics={
            "deepfake": keras.metrics.BinaryAccuracy(),
            "encoder": keras.metrics.CategoricalAccuracy(),
        },
    )
else:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics={"deepfake": keras.metrics.BinaryAccuracy()},
    )

scores = {}
sections = {s.strip().lower() for s in args.eval_sections.split(",") if s.strip()}

if "all" in sections and n_all > 0:
    print("\nALL:")
    steps_all = min(int(np.ceil((n_all * global_conf["repeat"]) / global_conf["batch_size"])), args.steps)
    scores["all"] = model.m.evaluate(it_test, batch_size=global_conf["batch_size"], steps=steps_all)
elif "all" in sections:
    print("\nALL: skipped (no samples)")

if "encoder" in sections:
    for enc in loader.encoders:
        print(f"\nENCODER {enc}:")
        fake_it, n_fake = loader.make_dataset(
            mode="encoder",
            encoder=enc,
            augmenter=augmenter,
            input_shape=input_shape,
            batch_size=global_conf["batch_size"],
            repeat=global_conf["repeat"],
        )
        if n_fake <= 0:
            print(f"ENCODER {enc}: skipped (no samples)")
            continue
        if not ENCODER:
            fake_it = fake_it.map(lambda x, y: (x, one_hot_encoder(y, model_n_encoders)))
        else:
            fake_it = fake_it.map(lambda x, y: (x, y[0]))
        if "use_raw" in global_conf:
            fake_it = fake_it.map(lambda x, y: (tf.expand_dims(x, -1), y))
        steps_enc = min(int(np.ceil((n_fake * global_conf["repeat"]) / global_conf["batch_size"])), args.steps)
        scores[enc] = model.m.evaluate(fake_it, batch_size=global_conf["batch_size"], steps=steps_enc)

if "real" in sections:
    print("\nREAL:")
    real_it, n_real = loader.make_dataset(
        mode="real",
        augmenter=augmenter,
        input_shape=input_shape,
        batch_size=global_conf["batch_size"],
        repeat=global_conf["repeat"],
    )
    if n_real > 0:
        real_it = real_it.map(lambda x, y: (x, one_hot_encoder(y, model_n_encoders)))
        if "use_raw" in global_conf:
            real_it = real_it.map(lambda x, y: (tf.expand_dims(x, -1), y))
        steps_real = min(int(np.ceil((n_real * global_conf["repeat"]) / global_conf["batch_size"])), args.steps)
        scores["real"] = model.m.evaluate(real_it, batch_size=global_conf["batch_size"], steps=steps_real)
    else:
        print("REAL: skipped (no real samples in manifest)")

# Save npy summary aligned with original behavior
out_npy = os.path.splitext(args.output_csv)[0] + ".npy"
np.save(out_npy, scores)
print(f"[Saved] scores npy: {out_npy}")

# Save per-file predictions for confusion-matrix tooling
pred_df = loader.dump_predictions(
    model=model.m,
    augmenter=augmenter,
    threshold=args.threshold,
    output_csv=args.output_csv,
    log_every=args.log_every,
)
print(f"[Saved] predictions csv: {args.output_csv}")

if len(pred_df):
    y_true = pred_df["target"].astype(int).values
    y_pred = pred_df["pred_label"].astype(int).values
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    print(f"[Confusion] TP={tp} TN={tn} FP={fp} FN={fn} ACC={acc:.4f}")
