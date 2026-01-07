# Imagle 🎯  
*A Wordle-inspired image similarity game powered by deep metric learning*

---

## Overview

**Imagle** is a daily visual guessing game inspired by *Wordle*, where players must identify which image best matches a given prompt image. Unlike traditional upload-based similarity demos, Imagle is a **closed-world retrieval game** built entirely on a curated dataset.

Each day:
- A **prompt image** is selected from the dataset
- Players are shown a pool of visually similar candidate images
- Exactly **one candidate** is the true match
- Players must reason visually and select the correct lookalike within a limited number of attempts

Under the hood, Imagle uses a **Siamese Neural Network** trained for image similarity to generate *hard negatives*, making each daily puzzle challenging but fair.

---

## Key Features

- 🧠 **Deep Metric Learning**  
  Siamese Neural Network trained on the TotallyLooksLike dataset to learn visual similarity

- 🧩 **Daily Puzzle Generation**  
  Deterministic “picture of the day” selection ensures everyone plays the same puzzle

- 🔍 **Hard Negative Mining**  
  Candidate pools are built using nearest-neighbor search in embedding space

- ⚡ **CPU-Friendly Inference**  
  All dataset embeddings are precomputed offline; daily puzzles are generated in milliseconds

- 🎮 **Wordle-Style Gameplay**  
  Limited attempts, visual feedback, and answer reveal mechanics

- 🖥️ **Modern Frontend**  
  React + Vite UI designed for fast iteration and clean presentation

---

## How It Works

### Dataset
Imagle uses the **TotallyLooksLike** dataset, which consists of paired images:

left/0000.jpg ↔ right/0000.jpg
left/0001.jpg ↔ right/0001.jpg

Each pair represents two different images that visually resemble one another.

---

### Model

- **Architecture:** Siamese Neural Network  
- **Backbone:** ResNet-based encoder  
- **Embedding Size:** 256  
- **Similarity Metric:** Cosine similarity  

The model learns to map images into an embedding space where visually similar images are close together.

---

### Offline Precomputation

To keep runtime fast and CPU-friendly:

- All `right/*.jpg` images are embedded **once offline**
- Embeddings are saved to disk (`.npy`)
- At runtime, only a single embedding is computed for the daily target

This allows:
- Fast startup
- No per-request ML overhead
- Simple brute-force cosine similarity over ~6k embeddings

---

### Daily Puzzle Generation

1. Select a deterministic daily target ID
2. Use the **correct right-image embedding** to find nearest neighbors
3. Select the top `K-1` most similar images as decoys
4. Insert the correct image and shuffle the pool
5. Serve:
   - Prompt image (`left/{id}.jpg`)
   - Candidate pool (`right/*.jpg`)

---

## Tech Stack

### Backend
- **Python**
- **FastAPI**
- **TensorFlow / Keras**
- **NumPy**
- **Pillow**

### Frontend
- **React**
- **Vite**
- **Axios**
- **CSS Grid / Flexbox**

---

## Project Structure

├── backend/
│ ├── app.py # FastAPI app & game logic
│ ├── comparison_model.keras # Trained Siamese model
│ ├── artifacts/
│ │ ├── right_embs.npy # Precomputed embeddings
│ │ └── ids.txt # Image IDs
│ └── scripts/
│ └── precompute_right_embs.py
│
├── data/
│ ├── left/
│ └── right/
│
├── frontend/
│ ├── src/
│ │ ├── App.tsx
│ │ └── App.css
│ └── vite.config.ts
│
└── README.md


---

## Running the Project Locally

### Prerequisites
- Python **3.9+**
- Node.js **18+**
- npm or yarn

---

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

If embeddings have not been precomputed yet:

```bash
python scripts/precompute_right_embs.py
```

Then start the server:

```bash
uvicorn app:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at:

```bash
http://localhost:5173
```

The frontend uses a Vite proxy to communicate with the backend, avoiding CORS issues during development.

