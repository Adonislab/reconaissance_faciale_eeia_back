# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.db import save_embedding, load_embedding
from torch.nn.functional import cosine_similarity
import torch
import os

app = FastAPI()
model = load_facenet_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    face_tensor = preprocess_face(image)
    embedding = get_embedding(model, face_tensor, device)
    save_embedding(name, embedding)
    return {"message": f"Embedding saved for {name}"}

@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image_from_bytes(await file1.read())
    img2 = read_image_from_bytes(await file2.read())

    emb1 = get_embedding(model, preprocess_face(img1), device)
    emb2 = get_embedding(model, preprocess_face(img2), device)

    sim = cosine_similarity(
        torch.tensor(emb1).unsqueeze(0),
        torch.tensor(emb2).unsqueeze(0)
    ).item()

    same = sim > 0.7
    return {"similarity": sim, "match": same}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image = read_image_from_bytes(await file.read())
    input_emb = get_embedding(model, preprocess_face(image), device)

    max_sim, best_match = 0, None
    for fname in os.listdir("embeddings"):
        name = fname.replace(".pt", "")
        emb = load_embedding(name)
        sim = cosine_similarity(
            torch.tensor(input_emb).unsqueeze(0),
            emb.unsqueeze(0)
        ).item()
        if sim > max_sim:
            max_sim = sim
            best_match = name

    return {"match": best_match, "similarity": max_sim}
