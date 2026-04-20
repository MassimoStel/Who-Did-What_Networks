"""
Minimal SVO validation utilities.
Provides dep-parsing-based SVO extraction and evaluation against a gold standard CSV.
"""
import pandas as pd
from .nlp_utils import get_spacy_nlp
from .svo_extraction import extract_svos, _passive_info


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
        An optional ``passive_approx`` column is ignored by this function.
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


def validate_passive(gold_csv_path, nlp=None, verbose=True):
    """
    Validate the full TEA pipeline's passive voice handling against a gold
    standard CSV that includes a ``passive_approx`` column.

    This function uses ``extract_svos()`` (not the minimal baseline) and
    checks that the ``passive_approx`` flag is correctly assigned for each
    sentence.

    Parameters
    ----------
    gold_csv_path : str
        Path to a CSV with columns: sentence, subject, verb, object, passive_approx.
    nlp : spacy.Language, optional
        A spaCy model. Loaded automatically if not provided.
    verbose : bool
        If True, print per-sentence PASS/FAIL results.

    Returns
    -------
    dict
        Summary with keys: total, passed, failed, accuracy, details (list of dicts).
    """
    if nlp is None:
        nlp = get_spacy_nlp()

    df_gold = pd.read_csv(gold_csv_path)

    # Ensure passive_approx column exists
    if "passive_approx" not in df_gold.columns:
        raise ValueError(
            "Gold CSV must contain a 'passive_approx' column. "
            "Use validate_svo() for CSVs without it."
        )

    results = []
    for _, row in df_gold.iterrows():
        sent = str(row["sentence"])
        if sent == "nan":
            continue

        gold_approx = int(row["passive_approx"])

        # Run full TEA pipeline
        doc = nlp(sent)
        df = extract_svos(doc)

        # Get Agent→Event rows
        agent_rows = df[(df["TEA"] == "Agent") & (df["TEA2"] == "Event")]

        if len(agent_rows) == 0:
            # No extraction — mark as fail if gold expected something
            got_approx = -1
        elif len(agent_rows) == 1:
            got_approx = int(agent_rows.iloc[0]["passive_approx"])
        else:
            # Multiple agent rows (e.g. coordinated): use max
            # (if any row is passive_approx=1, the sentence has passive approx)
            got_approx = int(agent_rows["passive_approx"].max())

        passed = got_approx == gold_approx
        results.append({
            "sentence": sent,
            "gold_approx": gold_approx,
            "got_approx": got_approx,
            "passed": passed,
        })

        if verbose:
            marker = "✓" if passed else "✗"
            print(f"  [{marker}] {sent}")
            if not passed:
                print(f"      passive_approx: got={got_approx} exp={gold_approx}")

    n_total = len(results)
    n_passed = sum(r["passed"] for r in results)
    n_failed = n_total - n_passed
    accuracy = n_passed / n_total if n_total else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"PASSIVE APPROX VALIDATION: {n_passed}/{n_total} "
              f"({accuracy:.1%} accuracy)")
        if n_failed:
            print(f"\nFailed sentences ({n_failed}):")
            for r in results:
                if not r["passed"]:
                    print(f"  - {r['sentence']}")
        print()

    return {
        "total": n_total,
        "passed": n_passed,
        "failed": n_failed,
        "accuracy": round(accuracy, 3),
        "details": results,
    }
