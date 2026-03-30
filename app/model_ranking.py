import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================
# CONFIGURATION DU MODELE RANKING
# ========================

RANKING_PATH = "model_ranking"
MAX_LEN_RANKING = 512

# Tokenizer et modèle
tokenizer_ranking = AutoTokenizer.from_pretrained(RANKING_PATH)
model_ranking = AutoModelForSequenceClassification.from_pretrained(RANKING_PATH)

# Device (GPU si dispo sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ranking.to(device)
model_ranking.eval()

# ========================
# FONCTION PREDICTION
# ========================
def predict_insertion(text, paragraph):
    """
    Retourne la meilleure position pour insérer un paragraphe selon le modèle de ranking
    """
    paragraphs = text.split("\n")
    best_idx = 0
    best_score = -float("inf")

    for i in range(len(paragraphs) + 1):
        prev_text = "\n".join(paragraphs[:i])
        next_text = "\n".join(paragraphs[i:])

        # Préparation input Ranking
        text_input = (
            str(prev_text)
            + f" {tokenizer_ranking.sep_token} "
            + str(paragraph)
            + f" {tokenizer_ranking.sep_token} "
            + str(next_text)
        )

        inputs = tokenizer_ranking(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN_RANKING,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_ranking(**inputs)

        score = outputs.logits.squeeze().item()

        if score > best_score:
            best_score = score
            best_idx = i

    paragraphs.insert(best_idx, paragraph)
    new_text = "\n".join(paragraphs)

    return {
        "position": best_idx,
        "score": best_score,
        "text_modified": new_text
    }