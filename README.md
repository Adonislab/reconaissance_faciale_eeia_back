
---

# README.md

````markdown
# Projet Reconnaissance Faciale Africaine avec Fine-tuning Facenet

lien : https://colab.research.google.com/drive/1ZimZAEjyPZ4eMzaju7fTbuT1GTlGBnyE#scrollTo=f_hUPJtfd6k7

## Description

Ce projet implémente un système de reconnaissance faciale spécialisé pour les visages africains, basé sur un modèle Facenet fine-tuné. Il inclut :

- Un modèle deep learning pour extraire des embeddings faciaux.
- Un pipeline d'extraction et pré-traitement des visages.
- Une API FastAPI pour la reconnaissance et la vérification des visages via embeddings.
- Gestion des données d'entraînement/test et fine-tuning du modèle.

---

## Fonctionnalités principales

- Extraction d’embeddings faciaux avec un modèle Facenet fine-tuné.
- Pipeline d’extraction de visages et création des embeddings.
- Reconnaissance faciale via comparaison d’embeddings.
- API REST pour interagir avec le système (upload, reconnaissance, gestion embeddings).
- Support GPU via PyTorch.

---

## Prérequis

- Python 3.8+
- PyTorch (compatible CUDA pour GPU recommandé)
- FastAPI
- Uvicorn
- torchvision
- numpy, Pillow
- autres dépendances dans `requirements.txt`

---

## Installation

1. Cloner le dépôt

```bash
git clone https://github.com/Adonislab/reconaissance_faciale_eeia_back.git
cd reconaissance_faciale_eeia_back
````

2. Créer et activer un environnement virtuel

```bash
python -m venv .env
source .env/bin/activate  # Linux/Mac
.env\Scripts\activate     # Windows
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. (Optionnel) Installer Git LFS si tu utilises des modèles volumineux

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

---

## Utilisation

### Entraînement du modèle

* Prépare tes dossiers `Training/` et `Test/` avec les images faciales extraites.
* Lance le script d’entraînement fine-tuning (ex: `train.py`)

```bash
python train.py
```

### Lancement de l’API FastAPI

```bash
uvicorn app.main:app --reload
```

L’API sera disponible sur `http://localhost:8000`

---

## Endpoints API (exemples)

* `POST /extract` : Upload une image, extrait les visages et crée embeddings.
* `POST /recognize` : Envoie un visage, reçoit la reconnaissance basée sur embeddings.
* `GET /embeddings` : Liste les embeddings stockés.

---

## Structure du projet

```
/app
  /models          # Modèles PyTorch (Facenet fine-tuned)
/app/main.py       # API FastAPI
/train.py          # Script de fine-tuning
/requirements.txt  # Librairies Python
.gitignore
README.md
```

---

## Conseils & bonnes pratiques

* Utilise un GPU pour accélérer l’entraînement.
* Assure-toi d’avoir assez d’images par classe pour former des triplets valides.
* Nettoie régulièrement le dossier `embeddings` pour éviter des données obsolètes.
* Sauvegarde régulièrement ton modèle `.pth` en local ou dans un stockage distant.

---

## Licence

[MIT License](LICENSE)

---

## Contact

NOBIME Tanguy Adonis
Email: [nobimetanguy19@gmail.com](mailto:nobimetanguy19@gmail.com)
GitHub: [Adonislab](https://github.com/Adonislab)

---

```
