from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.db import save_embedding, load_embedding
from app.face_detect import extract_faces
from torch.nn.functional import pairwise_distance
import torch
import os
import io

app = FastAPI()
model = load_facenet_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔧 Seuil pour décider si deux visages sont identiques
MATCH_THRESHOLD = 1.1


@app.get("/")
def read_root():
    return {"message": "Application de reconnaissance faciale"}


@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)

    faces = extract_faces(image)
    if not faces:
        raise HTTPException(status_code=400, detail="Aucun visage détecté.")

    face_tensor = preprocess_face(faces[0])
    embedding = get_embedding(model, face_tensor, device)
    save_embedding(name, embedding)
    return {"message": f"Embedding enregistré pour {name}."}


@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image_from_bytes(await file1.read())
    img2 = read_image_from_bytes(await file2.read())

    faces1 = extract_faces(img1)
    faces2 = extract_faces(img2)

    if not faces1 or not faces2:
        raise HTTPException(status_code=400, detail="Visage non détecté dans l'une ou les deux images.")

    emb1 = get_embedding(model, preprocess_face(faces1[0]), device)
    emb2 = get_embedding(model, preprocess_face(faces2[0]), device)

    distance = pairwise_distance(
        torch.tensor(emb1).unsqueeze(0),
        torch.tensor(emb2).unsqueeze(0)
    ).item()

    
    same = "les visages sont semblables" if distance < MATCH_THRESHOLD else "les visages ne sont pas semblables"
    return {"distance": distance, "match": same}


# ✅ FONCTION UTILE COMMUNE POUR LA RECONNAISSANCE
def find_best_match_by_distance(input_embedding, embeddings_dir="embeddings"):
    min_dist, best_match = float("inf"), None

    for fname in os.listdir(embeddings_dir):
        name = fname.replace(".pt", "")
        emb = load_embedding(name)
        dist = pairwise_distance(
            torch.tensor(input_embedding).unsqueeze(0),
            emb.unsqueeze(0)
        ).item()

        if dist < min_dist:
            min_dist = dist
            best_match = name

    return best_match, min_dist


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image = read_image_from_bytes(await file.read())
    faces = extract_faces(image)
    if not faces:
        raise HTTPException(status_code=400, detail="Aucun visage détecté.")

    input_emb = get_embedding(model, preprocess_face(faces[0]), device)

    best_match, distance = find_best_match_by_distance(input_emb)

    if distance > MATCH_THRESHOLD:
        return {"match": None, "message": "Je ne reconnais pas encore cette personne.", "distance": distance}

    return {"match": best_match, "distance": distance}


@app.post("/realtime_recognize")
async def realtime_recognize(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = read_image_from_bytes(image_bytes)

        faces = extract_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail="Aucun visage détecté.")

        face_tensor = preprocess_face(faces[0])
        input_emb = get_embedding(model, face_tensor, device)

        best_match, distance = find_best_match_by_distance(input_emb)

        if distance > MATCH_THRESHOLD:
            return JSONResponse(content={"identity": None, "message": "Je ne reconnais pas encore cette personne.", "distance": distance})

        return JSONResponse(content={"identity": best_match, "distance": distance})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
