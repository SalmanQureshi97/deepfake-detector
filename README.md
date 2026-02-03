# AI-music detection study

## Note about this fork and reproduced results

This repository is a **personal fork** of Deezer's `deepfake-detector` used for my own reproducibility experiments and for training new weights.

- **Why this fork exists**: while trying to reproduce the results from the paper *“AI-Generated Music Detection and its Challenges”*, I needed to (a) fix several reproducibility issues in the original codebase and (b) adapt the Musika autoencoder pipeline. The main technical changes I contributed upstream are:
  - Refactoring the Musika `encode_audio` pipeline and adding SavedModel support, as well as evaluation fixes ([commit `edc94ad`](https://github.com/deezer/deepfake-detector/commit/edc94ad04b721e4ba59ccfb25e14606ddaa4a78a)).
  - Reproducibility and path fixes, including consistent dataset paths, mono-to-stereo handling, and updated dataset creation scripts ([commit `53257d2`](https://github.com/deezer/deepfake-detector/commit/53257d2bdd7425578740060704a17eed99c195f1)).
  These changes have been merged into the **upstream** Deezer repository; this fork keeps them together with my own trained weights and logs.

- **New weights and logs in this fork**: I retrained the main `specnn_amplitude` detector using the updated code, and added the resulting SavedModel under `weights/final/specnn_amplitude/`. The evaluation outputs are saved in:
  - `results.txt`: initial evaluation runs with the original weights / configuration.
  - `results_after_retraining.txt`: evaluation after retraining with the updated setup.

- **Discrepancies w.r.t. the published numbers**: even after applying the official reproducibility fixes and retraining,
  the metrics reported in these text files do **not exactly match** all values published in the paper (e.g., some per-encoder accuracies diverge, and my first runs were noticeably below the reported scores before retraining improved them to be much closer to 99–100% on several encoders).

In short, this fork:
- Keeps the original research code,
- Incorporates my upstream reproducibility contributions,
- Adds my **own trained weights and evaluation logs** for transparency,
- But should **not** be read as an exact replica of the experimental setup that produced the tables in the paper.

---

Code repository of our research paper on AI-generated music detection ["AI-Generated Music Detection and its Challenges"](https://arxiv.org/pdf/2501.10111) - D. Afchar, G. Meseguer Brocal, R. Hennequin (accepted for IEEE ICASSP 2025).

We create an AI-music detector by detecting the use of an artificial decoder (e.g., a neural decoder). For that, we auto-encode a dataset of music with several such auto-encoders to train on. This setting enables us to avoid detecting confounding artefacts. For instance, if a dataset of artificial music only contains pop music, you don't want to inadvertently train a pop music detector. Here, the task is to distinguish real music from its reconstructed counterpart. With the same musical content and compression setting, only the autoencoder artefacts remain. We also verify that merely training on autoencoder allows the model to detect music fully-generated from prompts (i.e., not auto-encoded).

Examples of audio reconstructions may be found in the `audio_examples` folder or on the demo page: [research.deezer.com/deepfake-detector/](https://research.deezer.com/deepfake-detector/).

The FMA dataset is available at [github.com/mdeff/fma](https://github.com/mdeff/fma).

More than a detector, we ponder the larger consequences of deploying a detector: robustness to manipulation, generalisation to different models, interpretability, ...

⚠️ Following the recent press releases by Deezer on our [AI-music detection tool](https://newsroom-deezer.com/2025/01/deezer-deploys-cutting-edge-ai-detection-tool-for-music-streaming/), let us clarify something for interested readers: the tool available in this repository is **not** the tool we use in production for synthetic music detection. This is due to the delay between doing research and having a paper being published. Nevertheless, our new tool succeeds this present work, is elaborated by the same authors, and with the same concerns in mind, namely aiming for interpretability, almost perfect accuracy scores, and a focus on a possibility for recourse in case of false positives, generalisation to unknown scenarios and robustness to manipulation.

## License

We provide this repository under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. You may share (mirror) and adapt (borrow and alter) this content, providing that you credit this work and don't use it for commercial purposes.

## Cite

Either the ICASSP publication :
```
@inproceedings{afchar2025ai,
  title={AI-Generated Music Detection and its Challenges},
  author={Afchar, Darius and Meseguer-Brocal, Gabriel and Hennequin, Romain},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

or the previous Arxiv version (longer paper with more experiments on calibration and interpretability):
```
@article{afchar2024detecting,
  title={Detecting music deepfakes is easy but actually hard},
  author={Afchar, Darius and Meseguer-Brocal, Gabriel and Hennequin, Romain},
  journal={arXiv preprint arXiv:2405.04181},
  year={2024}
}
```

## Reproducibility instructions

To use the autoencoders, you need to clone the following repo into a `pretrained` folder :
* [Git Musika!](https://github.com/marcoppasini/musika)
* [LAC](https://github.com/hugofloresgarcia/lac), using the pretrained weights found in [VampNet](https://github.com/hugofloresgarcia/vampnet)

Then, in the `utils_encode.py` script in the musika folder, I added a method `encode_audio` to return latent from the model given an audio `wv`, namely copying the method `compress_whole_files` and doing a `return lat` instead of doing a `np.save` in the end.


