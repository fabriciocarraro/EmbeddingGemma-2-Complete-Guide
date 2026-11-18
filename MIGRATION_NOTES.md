# Migration notes: Gemini Embedding 2 → EmbeddingGemma 2

This notebook is a port of the [Gemini Embedding 2 — Complete Guide](https://github.com/fabriciocarraro/Gemini-Embedding-2-Complete-Guide) (Colab notebook, 120 cells). The section order, the assets (images, audios, videos, PDFs) and the experiments are the same. This file lists every place where the code or the explanation had to change, and why.

Sources for the EmbeddingGemma 2 facts: the [model card on Hugging Face](https://huggingface.co/google/embeddinggemma-2) (`google/embeddinggemma-2`, Apache 2.0, not gated), the model's `config_sentence_transformers.json` / `processor_config.json`, and the `sentence-transformers` 6.2 [input formats docs](https://github.com/huggingface/sentence-transformers/blob/v6.2.0/docs/input_formats.rst). Facts about Gemini Embedding 2 are the ones stated in the original guide.

## It is not a drop-in replacement

| Topic | Gemini Embedding 2 | EmbeddingGemma 2 | What the port does |
|---|---|---|---|
| Access | `google-genai` client + `GEMINI_API_KEY` | local weights via `sentence-transformers` | The API-key cell became a "load the model" cell (device + dtype). No secrets anywhere. |
| Precision | n/a (server side) | `bfloat16` or `float32`; **`float16` silently produces NaN / degraded vectors** | `bfloat16` on GPUs that support it, otherwise `float32`. |
| Dimensions | 3,072 (MRL down to 128) | 768 (MRL: 512, 256, 128) | FAISS index, storage math and the MRL section use 768 → 128. |
| Task types | not used in the original guide | text **prompt prefixes** recommended (`prompt_name=`) | `SentenceSimilarity` for word comparisons, `Clustering` for AG News, `SearchQuery` / `Document` for retrieval. Media (images, audio, video) take no prompt. |
| Images | `types.Part.from_bytes(..., mime_type="image/png")` | PIL image / path / URL | Images are decoded to PIL and passed directly. |
| Audio | MP3 bytes, native | waveform, **16 kHz mono** | `librosa.load(..., sr=16000, mono=True)` before embedding. (A file path also works; we keep the resampling explicit.) |
| Video | MP4 bytes, ≤120 s, ≤32 frames, audio ignored | frames at 1 fps, ≤32 frames by default, 140 tokens/frame, audio ignored | Frames decoded with **PyAV**, using the same sampling rule as the model's video processor, then passed with `do_sample_frames=False` and **280 tokens per frame** (see below). |
| PDF | native, ≤6 pages | **not supported** | Pages rendered with **PyMuPDF** and embedded together as images (one embedding per PDF); plain-text extraction shown as a comparison. |
| Combined inputs | `types.Content(parts=[...])` | dict `{"text": "... <|image|>", "image": [...]}` | Posts 2–4 use dict inputs; one embedding per post. |
| Context | 8,192 text tokens | 8,192 tokens shared by every modality in one input | Mentioned where it matters (posts, videos, PDFs). |

## Section by section

**Setup.** `!pip install faiss-cpu` became a Colab-only `%pip install` of `sentence-transformers>=6.1.0`, `transformers>=5.22.0`, `faiss-cpu`, `pymupdf` and `av`. `google.colab` is no longer imported unconditionally, so the notebook also runs locally. TensorFlow / `tensorflow_datasets` are no longer imported (they were only used for the Projector checkpoint). A `NOTEBOOK_HEADLESS=1` environment variable skips the interactive bits (TensorBoard server, file picker, view counter) for automated runs.

**Media files.** The 30 files the notebook uses (15 images, 6 audios, 5 videos, 4 PDFs) were copied from the original repository into this one, byte for byte (SHA-256 checked); the original `header.jpg` was not copied because it carries the Gemini Embedding 2 branding. The new `images/header.png` is the official EmbeddingGemma 2 banner from Google's model card (`ai.google.dev/gemma/images/embeddinggemma2_banner.png`, unmodified). `load_file()` reads a file from disk when it exists (a local clone) and otherwise downloads it from this repository's GitHub raw URL (`BASE_URL`), so the notebook no longer depends on the Gemini repository.

**View counter.** Kept, with a new key (`embeddinggemma-2-guide`), a 5 s timeout and a `try/except` so it can never break the notebook. ⚠️ Decide whether you want to keep it before publishing.

**Text.** Same words and same experiments. The first embedding is printed raw; everything after it uses task prompts, introduced in a new short subsection. The vocabulary text said "100 words"; the list has 70, so the text now says 70.

**Bug fix in "king − man + woman".** In the original, `exclude_source_words_1` was defined but never applied, so "king" itself came first in the table (0.84, with "queen" second at 0.72), while the text said the source words were excluded. The port applies the exclusion.

**Clustering.** Same AG News sample (25 per class, `random_state=42`), embedded with the `Clustering` prompt. The Projector files are written as TSV + `projector_config.pbtxt` instead of a TensorFlow checkpoint, so only `tensorboard` is needed. Added a static t-SNE plot and a nearest-neighbor accuracy number, because the TensorBoard view isn't saved in the notebook.

**Images.** Same 7 images. In the "image → words" experiment the vocabulary is embedded a second time with the `SearchQuery` prompt (text side of a cross-modal comparison). Displayed images are downscaled JPEG thumbnails: the original embedded the full 5–11 MB PNGs as base64 in the outputs.

**FAISS image search.** Same Tiny ImageNet sampling (5 per label, 100 images, 20 labels). `pil_to_bytes` is gone (the model takes PIL images); `get_image_embedding` batches on the GPU. The index has dimension 768. `show_results` no longer prints `samples[0]['label']` as if it were the query's label. The upload cell now works in three ways: Colab file picker, a local `MY_IMAGE_PATH`, or (default when headless / not in Colab) a held-out image from Tiny ImageNet's validation split, so the cell no longer fails with `IndexError` when nothing is uploaded (which is what the saved original run shows), including when "Cancel upload" is clicked in Colab.

**Combined embeddings.** Same 5 posts and same query. Post 2 (6 images) is `{"image": [6 PIL images]}` → one vector. Posts 3 and 4 are `{"text": post_text + "\n<|image|>", "image": image}` with the `Document` prompt; the query uses `SearchQuery`. The original downloaded the 6 carousel images twice; now `load_file()` is cached. A new cell re-runs the comparison with the symmetric `SentenceSimilarity` prompt, because the answer depends on the prompt (with it, the ranking is identical to Gemini's).

**Audio.** Same 6 files. Each MP3 is decoded and resampled to 16 kHz mono with `librosa` (the files are stereo at 24 kHz or 44.1 kHz). Text queries use `SearchQuery`.

**Video — why PyAV.** Passing a video path to `encode()` makes `transformers` decode it with `torchcodec`, which needs FFmpeg's *shared* libraries. On the Windows test machine, `torchcodec 0.17` failed to load (`libtorchcodec_core*.dll` could not find its FFmpeg dependencies; the system FFmpeg is a static build). Rather than asking readers to install FFmpeg system-wide, the notebook decodes the frames with PyAV (whose wheels bundle FFmpeg) using exactly the processor's own rule (`fps=1`, `max_frames=32`, uniform overflow — copied from `EmbeddingGemma2VideoProcessor.sample_frames`) and passes them with `do_sample_frames=False`. On Linux/Colab, passing the path directly should also work, but that path was **not executed** here.

**PDF.** Not a supported modality. Each PDF (5–6 pages) is rendered at 100 dpi and embedded as one dict input `{"image": pages}` (6 × 280 = 1,680 tokens, well within 8,192). The text-query experiment also shows the ranking with extracted text (`Document` prompt), for comparison. The "6 pages max" remark from Gemini was replaced by the token-budget explanation (~29 pages per input at 280 tokens/page).

**Dimensions (MRL).** 768/512/256/128 instead of 3,072…128. The benchmark table is the one in the EmbeddingGemma 2 model card. Instead of re-embedding the videos once per size (as the Gemini version did with `output_dimensionality`), the 768-d vectors are truncated and re-normalized; a check cell compares this with `encode(..., truncate_dim=128, normalize_embeddings=True)`.

**New sections.** "Task prompts" (text), "A quick look without TensorBoard" (clustering), a hubness check (audio), "Loading only the encoders you need" (text-only model, 270M parameters, compared against the full model), and a final recap table (expected answer vs. what the model returned, with margins, plus the Gemini guide's answer as a reference column when the inputs are identical).

## Changes made so the notebook stands on its own

The first version of the port kept every input identical to the Gemini guide, to compare the two models one-to-one. With those inputs, 12 of 16 experiments matched Gemini's top result. Since this notebook has to work on its own, a few things were then changed. Every alternative was declared before running, and all of them are listed below, including the ones that didn't work.

| Experiment | Before (Gemini-identical) | Change | After |
|---|---|---|---|
| All videos | 140 tokens/frame (default): text → video was a near-tie, Waves 0.620 vs Dog 0.614 | **280 tokens per frame**, a documented knob (`max_soft_tokens`) | Dog 0.652 vs Waves 0.578. Voice → video (Dog) and Surf → video (Waves) still right, with equal or larger margins. Tried and not adopted: 2 fps (Dog by only 0.008). |
| Text → audio | "What sound is associated with an emergency vehicle?" → wolf 0.685 vs ambulance 0.682 (tie) | Query **"Emergency vehicle siren"** | Ambulance 0.662 vs wolf 0.635. Tried and not adopted: "The siren of an emergency vehicle passing by" (tie, 0.700 vs 0.695), "Something is on fire and help is on the way" (rain wins), `QuestionAnswering` prompt (ambulance by 0.001). |
| Ambulance image → audio | rain 0.616 vs ambulance 0.576 | **Kept as is**, plus a new cell showing that rain is the closest sound to 5 of the 6 images (hubness) | Still a miss. Tried: 560 and 1,120 image tokens (rain still first), image + "What sound does this make?" (three-way tie), every vocabulary image → sounds and every sound → images (rain or a speech-to-"man" effect dominates), voice clips → images (both land on the man photo), and a simple hubness correction (fixes ambulance and dog → wolf, but with margins ≈0.013 and errors elsewhere, so not used). |
| Posts | judged against "Post 1" | Expected answer is now "Post 1 or Post 2" (both are about the Gemini AI model, which is what the query talks about) | Post 2 wins with the search prompt; the symmetric-prompt cell (Post 1 first) is still there. |
| Commentary | written as a comparison with Gemini | Rewritten to stand alone; Gemini appears only as occasional asides and as the reference column of the recap | 15 of 16 expected answers found. |

## Not carried over / caveats

- **Native PDF input** — replaced by page images (or text extraction). The two are not equivalent to Gemini's native PDF embedding.
- **Audio inside videos** — ignored by both models. Not a regression, but worth knowing.
- **Absolute scores are not comparable across models.** Different models have different similarity "baselines"; compare rankings, not raw cosine values.
- **Gemini numbers in the recap** come from the saved outputs of the original Colab run; they were not re-run (no API key was used in this port).
- **Photo → sound effect** is the weak spot in these tests (hubness of the rain clip). It's shown, not hidden.
- **`model.supports()` / `model.modalities` are not reliable with selective loading.** A model loaded with `config_kwargs={"vision_config": None, "audio_config": None}` still reports image/audio/video support (the processor is unchanged) and only fails, with a clear `ValueError`, when you pass an image.
