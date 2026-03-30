import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================
# CONFIGURATION DU MODÈLE ROBERTA
# ========================

ROBERTA_PATH = "model_camembert"
MAX_LEN_ROBERTA = 256

# Tokenizer et modèle
tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_PATH)
model_roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)

# Device (GPU si dispo sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_roberta.to(device)
model_roberta.eval()

# ========================
# FONCTION PREDICTION
# ========================
def predict_insertion(text, paragraph):
    """
    Retourne la meilleure position pour insérer un paragraphe selon nôtre model RoBERTa
    """
    paragraphs = text.split("\n")
    best_idx = 0
    best_score = -float("inf")

    for i in range(len(paragraphs) + 1):
        prev_text = "\n".join(paragraphs[:i])
        next_text = "\n".join(paragraphs[i:])

        # Préparation input RoBERTa
        inputs = tokenizer_roberta(
            text=str(paragraph),
            text_pair=str(prev_text) + f" {tokenizer_roberta.sep_token} " + str(next_text),
            return_tensors="pt",
            truncation="longest_first",
            max_length=MAX_LEN_ROBERTA,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_roberta(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()

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