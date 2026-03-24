import re
from .nlp_utils import get_stanza_nlp, get_spacy_nlp
from teanets.resources import _COREFERENCE_NOUNS


def text_preparation(text, clean=True, coref_solver="fastcoref"):
    """
    Prepares the text for further processing by cleaning it and resolving coreferences.

    Parameters:
    text (str): The input text to prepare.
    coref_solver (str): The coreference solver to use. Currently supports 'stanza' and 'fastcoref'. If None, coreferences are not resolved.

    Returns:
    str: The prepared text with coreferences resolved.
    """

    # Clean the text
    cleaned_text = clean_text(text)

    if coref_solver == None:
        return cleaned_text

    # Solve coreferences
    resolved_text = solve_coreferences(cleaned_text, coref_solver=coref_solver)

    return resolved_text


def clean_text(text):
    """
    Cleans a given text by removing double spaces and words inside parentheses.

    Parameters:
    text (str): The input string to clean.

    Returns:
    str: The cleaned text without double spaces and words inside parentheses.
    """

    # Step 1: Remove anything inside parentheses including the parentheses
    # The regex '\(.*?\)' matches any content inside parentheses and the parentheses themselves.
    cleaned_text = re.sub(r"\(.*?\)", "", text)
    cleaned_text = re.sub(r"\[.*?\]", "", text)

    # Step 2: Replace double spaces with single spaces
    # This step is repeated until all double spaces are reduced to single spaces
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    # Step 3: Strip leading and trailing spaces (if any)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def solve_coreferences(text, coref_solver="fastcoref"):
    """
    Resolves coreferences in a given text using either the Stanza library or fastcoref.

    Args:
        text (str): The input text to resolve coreferences in.
        coref_solver (str): The coreference solver to use. Currently supports 'stanza' and 'fastcoref'.
    Return:
        str: The text with coreferences resolved.
    """

    if coref_solver not in {"stanza", "fastcoref"}:
        raise ValueError(
            "Only stanza and fastcoref coreference solvers are supported at the moment."
        )

    if coref_solver == "stanza":
        # Load the Stanza pipeline
        stanzanlp = get_stanza_nlp()
        # Process the text
        doc = stanzanlp(text)
        output_text = stanza_solve_coreferences(doc)

    if coref_solver == "fastcoref":
        output_text = fastcoref_solve_coreferences(text)

    return output_text


def fastcoref_solve_coreferences(text_to_resolve):
    """
    Replaces coreferent mentions with their representative texts in the text reconstructed from the doc.

    Args:
        text.

    Returns:
        str: The text with coreferences resolved.
    """
    from fastcoref import FCoref as OriginalFCoref
    from transformers import AutoModel
    import logging

    import functools
    import re
    from tqdm import tqdm
    from functools import partial

    # Suppress transformers/fastcoref logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("fastcoref").setLevel(logging.ERROR)

    class PatchedFCoref(OriginalFCoref):
        def __init__(self, *args, **kwargs):
            original_from_config = AutoModel.from_config

            def patched_from_config(config, *args, **kwargs):
                kwargs["attn_implementation"] = "eager"
                return original_from_config(config, *args, **kwargs)

            try:
                AutoModel.from_config = functools.partial(
                    patched_from_config, attn_implementation="eager"
                )
                super().__init__(*args, **kwargs)
            finally:
                AutoModel.from_config = original_from_config

    model = PatchedFCoref(nlp=get_spacy_nlp(), device="cpu")

    preds = model.predict(texts=text_to_resolve)

    clusters_positions = preds.get_clusters(as_strings=False)
    clusters_strings = preds.get_clusters()
    text_to_resolve = text_to_resolve

    # Build a list of replacements
    replacements = []  # Each item: (start_pos, end_pos, replacement_text)

    for cluster_idx, cluster in enumerate(clusters_positions):
        mentions_positions = cluster  # List of (start_char, end_char)
        mentions_texts = clusters_strings[
            cluster_idx
        ]  # Corresponding list of mention texts

        # Build a list of mentions with their positions and texts
        mentions = []
        for pos, text_mention in zip(mentions_positions, mentions_texts):
            start, end = pos
            mention_text = text_mention
            mentions.append({"start": start, "end": end, "text": mention_text})

        # Identify mentions containing words in _COREFERENCE_NOUNS
        mentions_with_coref_nouns = []
        for mention in mentions:
            words_in_mention = re.findall(r"\b\w+\b", mention["text"].lower())
            if any(word in _COREFERENCE_NOUNS for word in words_in_mention):
                mentions_with_coref_nouns.append(mention)

        if mentions_with_coref_nouns:
            # Find a replacement mention that does not contain any word in _COREFERENCE_NOUNS
            replacement_mentions = []
            for m in mentions:
                words_in_mention = re.findall(r"\b\w+\b", m["text"].lower())
                if not any(word in _COREFERENCE_NOUNS for word in words_in_mention):
                    replacement_mentions.append(m)
            if not replacement_mentions:
                # If no replacement mention is available, skip this cluster
                continue
            # Prefer the earliest mention in the text
            replacement_mentions.sort(key=lambda m: (m["start"], -len(m["text"])))
            replacement_text = replacement_mentions[0]["text"]
            # For each mention with _COREFERENCE_NOUNS, record the replacement
            for mention in mentions_with_coref_nouns:
                replacements.append(
                    {
                        "start": mention["start"],
                        "end": mention["end"],
                        "replacement": replacement_text,
                    }
                )

    # Sort replacements in reverse order of start positions
    replacements.sort(key=lambda x: x["start"], reverse=True)

    # Apply replacements to the text
    for repl in replacements:
        start = repl["start"]
        end = repl["end"]
        replacement_text = repl["replacement"]
        text_to_resolve = (
            text_to_resolve[:start] + replacement_text + text_to_resolve[end:]
        )

    return text_to_resolve


def stanza_solve_coreferences(doc):
    """
    Replaces coreferent mentions with their representative texts in the text reconstructed from the doc.

    Args:
        doc (stanza.Document): The processed document with coreference information.

    Returns:
        str: The text with coreferences resolved.
    """
    # Use doc.text to get the original text
    original_text = doc.text

    # List to hold the replacements (start_char, end_char, representative_text)
    replacements = []

    # Dictionary to keep track of active mentions per coreference chain
    active_mentions = {}

    # Iterate over sentences and words to build mentions
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() not in _COREFERENCE_NOUNS:
                continue
            word_start = word.start_char
            word_end = word.end_char

            if word.coref_chains:
                # Get the coref_attachment with the lowest chain_idx
                min_coref_attachment = min(
                    word.coref_chains, key=lambda x: x.chain.index
                )
                chain_idx = min_coref_attachment.chain.index
                rep_text = min_coref_attachment.chain.representative_text

                if len(rep_text) > 15:
                    continue

                if word.text in rep_text:
                    continue

                # Start of a mention
                if min_coref_attachment.is_start:
                    active_mentions[chain_idx] = {
                        "start_char": word_start,
                        "end_char": word_end,
                        "rep_text": rep_text,
                    }
                # Continuation of a mention
                elif chain_idx in active_mentions:
                    active_mentions[chain_idx]["end_char"] = word_end

                # End of a mention
                if min_coref_attachment.is_end:
                    if chain_idx in active_mentions:
                        mention = active_mentions[chain_idx]
                        # Record the mention span and its representative text
                        replacements.append(
                            (
                                mention["start_char"],
                                mention["end_char"],
                                mention["rep_text"],
                            )
                        )
                        # Remove the mention from active mentions
                        del active_mentions[chain_idx]
            else:
                # Word is not part of any coreference chain; nothing to do
                pass

    # Sort replacements in reverse order to prevent index shifting
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements to the original text
    resolved_text = original_text
    for start_char, end_char, rep_text in replacements:
        resolved_text = resolved_text[:start_char] + rep_text + resolved_text[end_char:]

    return resolved_text
