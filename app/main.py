from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os

# ========================
# INITIALISATION APP
# ========================

app = FastAPI(title="Insertion de Paragraphe")

# ========================
# CONFIG TEMPLATES / STATIC
# ========================

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ========================
# DEBUG 
# ========================

print("Current working directory:", os.getcwd())

if os.path.exists("templates"):
    print("Templates trouvés :", os.listdir("templates"))
else:
    print("Dossier templates introuvable")

# ========================
# ROUTES PAGES
# ========================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/donnees", response_class=HTMLResponse)
async def donnees(request: Request):
    return templates.TemplateResponse("donnees.html", {"request": request})


@app.get("/scripts", response_class=HTMLResponse)
async def scripts(request: Request):
    return templates.TemplateResponse("scripts.html", {"request": request})


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})


@app.get("/essais", response_class=HTMLResponse)
async def essais(request: Request):
    """
    Affiche la page de test du modèle
    """
    return templates.TemplateResponse("essais.html", {
        "request": request,
        "text": "",
        "paragraph": "",
        "result": None,
        "model_choice": "roberta"
    })


# ========================
# ROUTE POST (TEST MODELE)
# ========================

@app.post("/essais", response_class=HTMLResponse)
async def essais_post(
    request: Request,
    text: str = Form(...),
    paragraph: str = Form(...),
    model_choice: str = Form("roberta")
):
    """
    Récupère les données du formulaire,
    appelle le modèle choisi,
    puis renvoie le résultat à la page HTML.
    """

    print("\n--- Nouvelle requête ---")
    print("Modèle choisi :", model_choice)
    print("Texte :", text[:100], "...")
    print("Paragraphe :", paragraph[:100], "...")

    # Choix du modèle selon la sélection
    if model_choice == "ranking":
        from app.model_ranking import predict_insertion as predict
    else:
        from app.model import predict_insertion as predict

    # Appel de la fonction avec 2 arguments uniquement
    result = predict(text, paragraph)

    print("Position prédite :", result["position"])
    print("Score :", result["score"])

    return templates.TemplateResponse("essais.html", {
        "request": request,
        "result": result,
        "text": text,
        "paragraph": paragraph,
        "model_choice": model_choice
    })


@app.get("/observations", response_class=HTMLResponse)
async def observations(request: Request):
    return templates.TemplateResponse("observations.html", {"request": request})


@app.get("/membres", response_class=HTMLResponse)
async def membres(request: Request):
    return templates.TemplateResponse("membres.html", {"request": request})