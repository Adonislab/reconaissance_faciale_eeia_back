from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def extract_faces(image, return_multiple=False):
    """
    Détecte les visages dans une image (PIL ou np.ndarray) et retourne les visages recadrés.
    """
    # Si image est un tableau NumPy, convertis-le en image PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        return []

    faces = []
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face = image.crop((x1, y1, x2, y2))
        faces.append(face)

    return faces if return_multiple else faces[:1]
