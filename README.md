![Header](https://raw.githubusercontent.com/fabriciocarraro/EmbeddingGemma-2-Complete-Guide/refs/heads/main/images/header.png)

# EmbeddingGemma 2 — Complete Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U-qs9RqjAHjPM0xuQsRCTlAcHLbmCpMe?usp=sharing)

**EmbeddingGemma 2** is Google DeepMind's open multimodal embedding model. It maps text (including code), images, audio, and video into the same 768-dimensional vector space, enabling cross-modal search through simple cosine similarity, with a model small enough (740M parameters) to run on your own machine.

---
 
## What is this?
 
A hands-on notebook that walks you through **everything** EmbeddingGemma 2 can do, from classic word-vector arithmetic ("king − man + woman = queen") to fully cross-modal searches where a voice clip finds a video, or a text query finds a sound. It's the open-model version of our [Gemini Embedding 2 — Complete Guide](https://github.com/fabriciocarraro/Gemini-Embedding-2-Complete-Guide), and it points out every place where the two models behave differently.
 
Everything runs in a single Google Colab. No API key needed: the model is open (Apache 2.0) and its weights are downloaded from Hugging Face on the first run.

---

## Authors
 
- **Fabrício Carraro** - [LinkedIn](https://www.linkedin.com/in/fabriciocarraro/) - fabriciocarraro@gmail.com
- **Pedro Gengo** - [LinkedIn](https://www.linkedin.com/in/pedrogengo/) - pedro.gengo.lourenco@gmail.com
 
---

## What's Inside

**📝 Text Embeddings** — What embeddings are, task prompts, cosine similarity, and the classic "king − man + woman = queen" experiment. Plus text clustering with TensorBoard.

**🖼️ Image Embeddings** — Image-to-text matching, image-to-image comparison, image embedding arithmetic (king − crown = man).

**🔍 Image Search** — A full similarity search engine built with FAISS and Tiny ImageNet. Upload your own image and find the closest matches.

**🧩 Combined Multimodal Embeddings** — Aggregating text + images into single embeddings for social media posts, then searching across 5 posts that all mention "Gemini" in different contexts (AI model, zodiac sign, NASA mission).

**🔊 Audio Embeddings** — Text-to-audio search, audio-to-audio matching, and image-to-audio cross-modal search.

**🎬 Video Embeddings** — Text-to-video, audio-to-video, and video-to-video similarity search, with a look at the exact frames the model sees.

**📄 PDF file Embeddings** — PDFs aren't a native input of EmbeddingGemma 2, so we render their pages as images: text-to-PDF search, audio-to-PDF cross-modal search, and PDF-to-PDF comparison.

**📐 Dimensionality** — Testing how Matryoshka Representation Learning (MRL) lets you reduce embedding dimensions from 768 down to 128.

**🪶 Selective loading** — Loading only the text part of the model (270M parameters) when you don't need images or audio.

---

## Getting Started

1. Open the notebook in Google Colab using the badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U-qs9RqjAHjPM0xuQsRCTlAcHLbmCpMe?usp=sharing)
2. Choose a GPU runtime: *Runtime → Change runtime type → T4 GPU*
3. Run the cells and explore

---

## Dependencies
 
The notebook installs everything it needs automatically. Key libraries:
 
| Library | Purpose |
|---|---|
| `sentence-transformers` / `transformers` | Loading and running EmbeddingGemma 2 |
| `torch` | Inference on the GPU |
| `faiss-cpu` | Fast similarity search (FAISS) |
| `librosa` | Decoding audio and resampling it to 16 kHz |
| `av` | Decoding video frames |
| `pymupdf` | Rendering PDF pages as images |
| `scikit-learn` | Cosine similarity |
| `tensorboard` | Embedding visualization (Projector) |
| `datasets` | Hugging Face dataset loading |
| `matplotlib` / `Pillow` | Image display and plotting |
| `pandas` / `numpy` | Data manipulation |

---
