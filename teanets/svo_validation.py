"""
Minimal SVO validation utilities.
Provides dep-parsing-based SVO extraction and evaluation against a gold standard CSV.
"""
import pandas as pd
from .nlp_utils import get_spacy_nlp


def _norm(x):
    """Normalise a string for comparison."""
    s = str(x).strip().lower()
    return "__none__" if s in {"", "nan", "none"} else s


def _prf(pred, gold):
    """Compute Precision, Recall, F1 for two Series."""
    p = pred.apply(_norm)
    g = gold.apply(_norm)
    tp = (p == g).sum()
    n_pred = (p != "__none__").sum()
    n_gold = (g != "__none__").sum()
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": tp, "Pred": n_pred, "Gold": n_gold,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
    }


def extract_svo_dep(doc):
    """
    Extract a single (subject, verb, object) triple from a spaCy Doc
    using pure dependency parsing (ROOT + nsubj/nsubjpass + dobj/pobj/...).
    """
    subj = verb = obj = None
    for token in doc:
        if token.dep_ == "ROOT":
            verb = token.lemma_
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "nsubj:pass",
                                  "csubj", "csubjpass"):
                    subj = child.lemma_
                if child.dep_ in ("dobj", "pobj", "attr", "acomp", "ccomp"):
                    obj = child.lemma_
                elif child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = f"{child.lemma_} {gc.lemma_}"
                elif child.dep_ == "agent":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = f"by {gc.lemma_}"
                elif child.dep_ == "xcomp":
                    for gc in child.children:
                        if gc.dep_ in ("dobj", "pobj", "attr"):
                            obj = gc.lemma_
            break
    return subj, verb, obj


def validate_svo(gold_csv_path, nlp=None):
    """
    Validate dep-parsing SVO extraction against a gold standard CSV.

    Parameters
    ----------
    gold_csv_path : str
        Path to a CSV with columns: sentence, subject, verb, object.
    nlp : spacy.Language, optional
        A spaCy model. Loaded automatically if not provided.

    Returns
    -------
    pd.DataFrame
        A DataFrame with Precision, Recall, F1 for Subject, Verb, Object.
    """
    if nlp is None:
        nlp = get_spacy_nlp()

    df_gold = pd.read_csv(gold_csv_path)
    rows = []
    for i, row in df_gold.iterrows():
        s = str(row["sentence"])
        if s == "nan":
            continue
        subj, verb, obj = extract_svo_dep(nlp(s))
        rows.append({
            "pred_subj": subj, "pred_verb": verb, "pred_obj": obj,
            "gold_subject": row["subject"],
            "gold_verb": row["verb"],
            "gold_object": row["object"],
        })

    df_eval = pd.DataFrame(rows)
    metrics = {
        "Subject": _prf(df_eval["pred_subj"], df_eval["gold_subject"]),
        "Verb":    _prf(df_eval["pred_verb"], df_eval["gold_verb"]),
        "Object":  _prf(df_eval["pred_obj"],  df_eval["gold_object"]),
    }
    return pd.DataFrame(metrics).T
