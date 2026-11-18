# Test report

What was actually executed to validate this port, what was replaced by a stand-in, and what was not run. Date of the runs: 2026-10-29/30, with a re-check on 2026-11-18.

## Environment

| | |
|---|---|
| OS | Windows 11 Pro (10.0.26300) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM (driver 591.74, CUDA 13.1) |
| Python | 3.12.9, isolated virtual environment |
| Key packages | torch 2.14.1+cu130, transformers 5.22.0, sentence-transformers 6.2.0, faiss-cpu 1.15.1, librosa 1.0.0, av 19.0.2, pymupdf 2.0, tensorboard 2.21.0, datasets 5.1.0, pandas 3.1.0, numpy 2.5.4 |
| Model | `google/embeddinggemma-2` from Hugging Face (Apache 2.0, not gated), `bfloat16` on GPU |
| Media files | the 30 files in this repository (`images/`, `audios/`, `videos/`, `PDFs/`), copied byte for byte from the Gemini guide's repository (SHA-256 checked) and read from disk by the final run |

## ✅ Executed (real runs, real model inference)

| # | Test | Result |
|---|---|---|
| 1 | **Whole notebook, top to bottom**, headless (`NOTEBOOK_HEADLESS=1`, `nbclient`, fresh kernel), run from inside the repository folder (as after a `git clone`), in an environment installed from `requirements.txt` | 63/63 code cells executed in order, **0 errors**, 189 s, media read from disk (2026-10-30). |
| 1a | Final version (header image + upload cell that survives "Cancel upload"), re-run on 2026-11-18 from the repository folder with the latest library releases | 63/63 code cells, **0 errors**, 130 s, every number identical to the previous runs. This is the run saved in the published `.ipynb`. |
| 1c | **Simulated** (not Colab): the "Search with your own image" cell with `google.colab.files` replaced by a stub returning `{}` (what Colab returns on "Cancel upload") or one uploaded file | Cancel → falls back to the validation image; upload → uses the uploaded image; outside Colab → fallback. The real Colab file picker was not exercised. |
| 1b | Same notebook re-run on 2026-11-18 with the latest releases at that date (sentence-transformers 6.3.0, transformers 5.23.0, faiss-cpu 1.16.0, librosa 1.0.1, pymupdf 2.1, datasets 5.2.0; torch 2.14.1+cu130 is still the newest CUDA 13.0 build). Model revision on Hugging Face unchanged since 2026-10-28. | 63/63 code cells, **0 errors**, 181 s, every number identical to the published run. |
| 2 | Earlier full runs: same notebook with media downloaded from GitHub (63/63, 0 errors, twice), and two runs of the first, Gemini-identical version (61/61 and 62/62, 0 errors) | Every number shared between runs of the same version was identical (deterministic on this machine), whether the media came from disk or from GitHub. |
| 2b | `load_file()` from the notebook, executed in a folder without the media files, i.e. the Colab path | Downloaded all 30 files; all 30 are byte-identical to the copies in this repository. Pointed at the Gemini repository for this test (same files), because this repository doesn't exist on GitHub yet. |
| 2c | `pip install -r requirements.txt` into a fresh venv (after the CUDA build of torch) | Installed without errors; kept `torch 2.14.1+cu130`; did not pull in `torchcodec`. |
| 3 | API probe of EmbeddingGemma 2 + sentence-transformers | 768-d output, 744M parameters, max 2.03 GB VRAM. `prompt_name` also applies to the text of dict inputs (cos = 1.0 vs. a manual prefix). Several images in one dict → one embedding. MP3 path vs. our 16 kHz waveform: cos = 1.0. Video frames with vs. without metadata: cos = 1.0. |
| 4 | TensorBoard Projector, against the files the notebook wrote | Server started, Projector plugin active, serves the 100 × 768 float32 tensor (307,200 bytes) and 100 metadata rows + header. Checked through TensorBoard's HTTP API, not by looking at the UI. |
| 5 | Sensitivity of the 4 experiments that disagreed with Gemini (first version) to prompt choice and precision (`bfloat16` vs. `float32`, both on GPU) | Precision changes scores by ≤ 0.003. Posts: ranking depends on the prompt (symmetric prompt reproduces Gemini's order exactly). Text → audio: wolf/ambulance within 0.003 in every setting; the order flips with the setting. Image → audio: rain first in every setting. Text → video: Waves/Dog close; Dog first without a prompt. |
| 5b | Pre-declared fixes for those experiments (vision token budget 560/1,120, image + question query, `QuestionAnswering` prompt, 280 tokens per video frame, 2 fps, 3 alternative audio queries, all image ↔ sound pairs, voice clips → images, a hubness correction) | Adopted: 280 tokens per frame (text → video fixed, other video results kept) and the query "Emergency vehicle siren". Nothing fixed photo → sound effect. Every outcome is listed in MIGRATION_NOTES.md. |
| 6 | Text-only loading (`config_kwargs`) | 271M parameters, +0.54 GB VRAM, same text embedding as the full model (cos 1.0000), image input rejected with `ValueError`; `supports("image")` still returns `True`. |
| 7 | Manual MRL truncation vs. `truncate_dim=128, normalize_embeddings=True` | Max abs. difference 0.0013 (`bfloat16` rounding). |
| 8 | Visual check of the HTML outputs (thumbnails, rankings, 61 video frames, 23 PDF pages) | Decoded from the notebook and inspected: correct images, in the ranked order, with their scores. |
| 9 | Notebook hygiene | `nbformat` validation passes; scanned all sources and outputs for local paths and secrets. One stderr warning (tqdm/ipywidgets) containing a local Windows path was removed from the published file; nothing else found. No API keys or tokens are used or stored anywhere. |

## 🔁 Replaced by a stand-in (so the notebook could run unattended)

| What | Stand-in | Why |
|---|---|---|
| Uploading your own image to the FAISS search (`files.upload()`) | First Tiny ImageNet **validation** image whose class is in the index (a goldfish, not in the index) | No Colab file picker in a headless run. The branch for `MY_IMAGE_PATH` / Colab upload was not executed. |
| `%tensorboard` inline server | Skipped in headless mode; server tested separately (test 4) | A headless run can't show an interactive iframe. |
| Video decoding by the model's own processor (via `torchcodec`) | Frames decoded with PyAV, same sampling rule, passed with `do_sample_frames=False` | See "Blocked" below. Parity between our frames and `torchcodec`'s was **not** verified (no working `torchcodec` here). |

No result in the notebook or in this report is simulated or hand-written: every score comes from a model run.

## ⏸️ Not executed

| What | Status | Note |
|---|---|---|
| Running the notebook in **Google Colab** | **Pending** (needs the repository on GitHub and a person with a Colab session) | The Colab-only `%pip install` cell and the `google.colab` upload branch never ran. In Colab the media are downloaded from `BASE_URL` (`fabriciocarraro/EmbeddingGemma-2-Complete-Guide`), which returns HTTP 404 until that repository is pushed; if it gets another name, update `BASE_URL` and the README badge. Recommended before publishing: open it in Colab with a T4 runtime and *Run all*. On a T4, `bfloat16` is emulated: it should work, but speed was not measured. |
| Running on **CPU only** | Not executed | Expected to work in `float32`, slower (especially images/videos). |
| The view counter request (`countapi.mileshilliard.com`, new key `embeddinggemma-2-guide`) | Skipped on purpose (`HEADLESS`) | To avoid registering test views. Decide whether to keep it before publishing. |
| Gemini Embedding 2 numbers | Not re-run | Taken from the saved outputs of the original Colab notebook. No Gemini API key was used. |

## ⛔ Blocked

| Action | Target | Reason | Result obtained | Status |
|---|---|---|---|---|
| Decode videos (and audio) through `torchcodec`, the default backend `transformers` uses for video paths | `torchcodec 0.17.0` on Windows 11 | `libtorchcodec_core{4..7}.dll` could not load: FFmpeg *shared* libraries are not installed (the system FFmpeg is a static build). Installing FFmpeg system-wide was not done. | `OSError` on import; `torchcodec` uninstalled from the test environment; video handled with PyAV (all video cells pass) and audio with librosa. | Worked around, not resolved. Unverified whether the path-based route works in Colab/Linux. |

## Results summary (from the published run)

The expected answer comes out on top in 15 of 16 experiments. The miss is ambulance image → audio (rain instead of ambulance, hubness). Full table in the README and at the end of the notebook; how the first, Gemini-identical version did (12 of 16 matching Gemini) is in MIGRATION_NOTES.md.
