from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import timm
import torch
import faiss
import requests
import numpy as np
import json
import glob
import os
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Initialize Models & FAISS ---
app = FastAPI()
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Placeholder for your master card embeddings and IDs
# In a real app, load this from a file or database
card_embeddings = [] 
card_ids = [] 
index = None
DIMENSION = 512 # CLIP-base embedding dimension

def build_faiss_index(embeddings):
    global index
    if embeddings:
        index = faiss.IndexFlatL2(DIMENSION)
        index.add(torch.tensor(embeddings, dtype=torch.float32).cpu().numpy())
        print(f"FAISS index built with {len(embeddings)} vectors.")
    else:
        print("No embeddings to build FAISS index.")

# --- API Endpoints ---


def build_condition_index():
    """Loads labeled condition dataset from web/test/fixtures/condition_dataset and builds a FAISS index."""
    global condition_embeddings, condition_ids, condition_labels, condition_index

    base_dir = os.path.join(os.path.dirname(__file__), 'web', 'test', 'fixtures', 'condition_dataset')
    labels_path = os.path.join(base_dir, 'labels.json')

    if not os.path.exists(labels_path):
        print("No condition dataset labels found at:", labels_path)
        return

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    embeddings = []
    ids = []
    labs = []

    for item in labels:
        img_rel = item.get('image')
        img_path = os.path.join(base_dir, img_rel)
        if not os.path.exists(img_path):
            print(f"Missing image for condition dataset: {img_path}, skipping")
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            inputs = processor(images=img, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            vec = feats.cpu().numpy().astype('float32').reshape(-1)
            embeddings.append(vec)
            ids.append(item['id'])
            labs.append(item['label'])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    if embeddings:
        condition_embeddings = np.vstack(embeddings)
        condition_ids = ids
        condition_labels = labs
        condition_index = faiss.IndexFlatL2(DIMENSION)
        condition_index.add(condition_embeddings)
        print(f"Condition FAISS index built with {condition_index.ntotal} vectors.")
    else:
        print("No embeddings generated for condition dataset.")


@app.post('/condition')
async def infer_condition(file: UploadFile = File(...)):
    """Infer condition probabilities using nearest neighbors in the condition dataset.
    Returns: { probabilities: {mint:0.1,...}, predicted: 'nm', confidence: 0.87 }
    """
    global condition_index, condition_labels, condition_ids

    try:
        if condition_index is None or condition_index.ntotal == 0:
            return JSONResponse(content={"error": "No condition dataset index available"}, status_code=503)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        inputs = processor(images=image, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        vec = feats.cpu().numpy().astype('float32')

        # k nearest neighbors
        k = min(5, condition_index.ntotal)
        distances, indices = condition_index.search(vec, k)
        distances = distances[0]
        indices = indices[0]

        # Convert distances to similarity scores (smaller distance -> higher score)
        # Use simple negative distance and softmax
        sims = np.exp(-distances)
        if sims.sum() == 0:
            probs = {}
        else:
            # Aggregate probabilities per class
            class_scores = {}
            class_counts = {}
            for dist, idx in zip(distances, indices):
                lab = condition_labels[idx]
                score = float(np.exp(-dist))
                class_scores[lab] = class_scores.get(lab, 0.0) + score
                class_counts[lab] = class_counts.get(lab, 0) + 1

            total = sum(class_scores.values())
            probs = {k: float(v / total) for k, v in class_scores.items()}

        # Ensure all classes present
        classes = ['mint', 'nm', 'lp', 'mp', 'hp', 'damaged']
        for c in classes:
            probs.setdefault(c, 0.0)

        predicted = max(probs.items(), key=lambda x: x[1])[0]
        confidence = float(probs[predicted])

        return JSONResponse(content={"probabilities": probs, "predicted": predicted, "confidence": confidence})

    except Exception as e:
        print('Condition inference error:', e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/scan")
async def scan_card(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 1. Get Image Embedding
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        query_vector = image_features.cpu().numpy()
        
        # 2. Search FAISS for nearest neighbors
        if index and index.ntotal > 0:
            distances, indices = index.search(query_vector, k=3) # Top 3 candidates
            
            candidates = []
            for i in range(len(indices[0])):
                candidate_id = card_ids[indices[0][i]]
                distance = distances[0][i]
                candidates.append({"card_id": candidate_id, "distance": float(distance)})
            
            return JSONResponse(content={"candidates": candidates})
        else:
            return JSONResponse(content={"message": "No card index available for search."}, status_code=500)

    except Exception as e:
        print(f"Error during scan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Server Startup ---
if __name__ == "__main__":
    # In a real app, you would load your pre-computed embeddings here
    # build_faiss_index(loaded_embeddings)
    
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
