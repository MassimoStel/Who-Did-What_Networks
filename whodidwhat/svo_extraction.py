from whodidwhat.textloader import text_preparation
from whodidwhat.WDWplot import *
from whodidwhat.resources import _VAGUE_ADVMODS, _VAGUE_AUX, _VAGUE_ADJ
import spacy
import spacy_transformers
from .nlp_utils import spacynlp, compute_valence
from itertools import combinations, chain
from nltk.corpus import wordnet as wn
import pandas as pd


def extract_svos_from_text(text, coref_solver="fastcoref"):
    """
    Extract Subject-Verb-Object (SVO) triples from a given text.
    """
    prepared_text = text_preparation(text, coref_solver=coref_solver)
    doc = spacynlp(prepared_text)
    svos = extract_svos(doc)
    return svos


################################################################################################
## Extract svos code.
################################################################################################


def get_synsets(phrase):
    """
    Get noun synsets for the phrase as a whole.
    """
    # Attempt to get synsets for the whole phrase as a noun
    synsets = wn.synsets(phrase.replace(" ", "_"), pos="n")
    if synsets:
        return synsets
    else:
        # Split the phrase into words and get noun synsets for each noun word
        words = phrase.split()
        synsets = []
        for word in words:
            # Get noun synsets for each word
            synsets.extend(wn.synsets(word, pos="n"))
        return synsets


def are_synonymous(word1, word2):
    """
    Check if two words or phrases are synonyms considering only nouns.
    Enforces the rule: if a word contains two nouns, return False as synonyms.
    """
    # Check if either word contains two or more nouns
    for phrase in [word1, word2]:
        words = phrase.split()
        noun_count = 0
        for word in words:
            # Check if the word has noun synsets
            if wn.synsets(word, pos="n"):
                noun_count += 1
        if noun_count >= 2:
            return False

    synsets1 = get_synsets(word1)
    synsets2 = get_synsets(word2)
    for syn1 in synsets1:
        for syn2 in synsets2:
            # Check if synsets are the same
            if syn1 == syn2:
                return True
            # Check if they share any lemma names (nouns only)
            if set(syn1.lemma_names()).intersection(set(syn2.lemma_names())):
                return True
    return False


def extract_svos(doc):
    """
    Extract Subject-Verb-Object (SVO) triples from a parsed document.
    Returns a pandas DataFrame with specified columns.
    """
    # Collect SVO triples
    svo_triples = []

    for possible_verb in doc:
        if (
            (possible_verb.pos_ in ("VERB", "AUX"))
            and (
                possible_verb.dep_
                not in ("aux", "auxpass", "amod", "npadvmod", "prep", "xcomp", "csubj")
            )
            and not (
                possible_verb.dep_ == "conj"
                and possible_verb.head.dep_
                in ("aux", "auxpass", "amod", "prep", "xcomp", "csubj")
            )
        ):

            subjects = get_verb_subjects(possible_verb)
            objects = get_verb_objects(possible_verb)
            verb_phrase = get_verb_phrase(possible_verb)

            svo_triples.append([subjects, verb_phrase, objects])

    # Initialize list to hold DataFrame rows
    data_rows = []

    # Process SVO triples to create DataFrame rows
    svo_id = 0
    for svo in svo_triples:
        subjects, verbs, objects = svo
        hypergraph = str(svo)

        # For syntactic relations (Semantic-Syntactic = 0)
        # Create Subject-Verb relations
        for subj, _ in subjects:
            for verb in verbs:
                # Subject - Verb relation
                data_rows.append(
                    {
                        "Node 1": subj,
                        "WDW": "Who",
                        "Node 2": verb,
                        "WDW2": "Did",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                    }
                )

        # Create Verb-Object relations
        for verb in verbs:
            for obj, _ in objects:
                # Verb - Object relation
                data_rows.append(
                    {
                        "Node 1": verb,
                        "WDW": "Did",
                        "Node 2": obj,
                        "WDW2": "What",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                    }
                )

        # Create Subject-Subject relations when multiple subjects are present
        if len(subjects) > 1:
            for (subj1, _), (subj2, _) in combinations(subjects, 2):
                data_rows.append(
                    {
                        "Node 1": subj1,
                        "WDW": "Who",
                        "Node 2": subj2,
                        "WDW2": "Who",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                    }
                )

        # Create Object-Object relations when multiple objects are present
        if len(objects) > 1:
            for (obj1, _), (obj2, _) in combinations(objects, 2):
                data_rows.append(
                    {
                        "Node 1": obj1,
                        "WDW": "What",
                        "Node 2": obj2,
                        "WDW2": "What",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                    }
                )
        svo_id += 1

    # Now, process for semantic relations (Semantic-Syntactic = 1)
    # Collect all subjects and objects across all SVOs
    all_subjects = [subj for svo in svo_triples for subj, _ in svo[0]]
    all_objects = [obj for svo in svo_triples for obj, _ in svo[2]]

    # Remove duplicates
    subject_nodes = list(set(all_subjects))
    object_nodes = list(set(all_objects))

    # For each pair of subjects, check if they are synonyms
    subject_pairs = combinations(subject_nodes, 2)
    for subj1, subj2 in subject_pairs:
        if are_synonymous(subj1, subj2):
            # Add semantic relation
            data_rows.append(
                {
                    "Node 1": subj1,
                    "WDW": "Who",
                    "Node 2": subj2,
                    "WDW2": "Who",
                    "Hypergraph": "N/A",
                    "Semantic-Syntactic": 1,
                    "svo_id": "N/A",
                }
            )

    # Similarly for objects
    object_pairs = combinations(object_nodes, 2)
    for obj1, obj2 in object_pairs:
        if are_synonymous(obj1, obj2):
            data_rows.append(
                {
                    "Node 1": obj1,
                    "WDW": "What",
                    "Node 2": obj2,
                    "WDW2": "What",
                    "Hypergraph": "N/A",
                    "Semantic-Syntactic": 1,
                    "svo_id": "N/A",
                }
            )

    # Create DataFrame
    df = pd.DataFrame(
        data_rows,
        columns=[
            "Node 1",
            "WDW",
            "Node 2",
            "WDW2",
            "Hypergraph",
            "Semantic-Syntactic",
            "svo_id",
        ],
    )

    return df


def get_verb_phrase(verb):
    """
    Retrieve the verb and its modifiers (negations, adverbs) as a single string.
    Also includes auxiliary verbs and connected verbs (xcomp, ccomp, conj).
    """
    parts = []

    # Get auxiliaries and negations before the verb
    aux = [
        child.lemma_
        for child in verb.lefts
        if child.dep_ in {"aux", "auxpass", "neg"} and child.lemma_ not in _VAGUE_AUX
    ]
    parts.extend(aux)

    ## Get adverbial modifiers before the verb
    # adverbs_before = [child.lemma_ for child in verb.lefts if child.dep_ in {'advmod', 'amod','npadvmod'} and \
    #                   child.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and child.lemma_ not in _VAGUE_ADVMODS]
    # parts.extend(adverbs_before)
    # Get adverbial before after the verb
    for child in verb.lefts:
        if (
            child.dep_ in {"advmod", "amod", "npadvmod"}
            and child.pos_ not in {"SCONJ", "CCONJ", "PART", "DET"}
            and child.lemma_ not in _VAGUE_ADVMODS
        ):
            # Include the noun and its modifiers
            noun_phrase, prep_phrases = get_compound_parts(child)
            parts.append(noun_phrase)

    # Add the main verb
    parts.append(verb.lemma_)

    ## In get_verb_phrase(verb), after processing adverbs_after
    # for child in verb.children:
    #    if child.dep_ == 'acomp':
    #        # Include the 'acomp' adjective in the verb phrase
    #        parts.append(child.lemma_)
    #        # Include any modifiers of the 'acomp' adjective
    #        for grandchild in child.children:
    #            if grandchild.dep_ in {'advmod', 'amod', 'npadvmod'} and\
    #                grandchild.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and\
    #                    grandchild.lemma_ not in VAGUE_ADVMODS:
    #                parts.append(grandchild.lemma_)

    # Get adverbial modifiers after the verb
    # adverbs_after = [child.lemma_ for child in verb.rights if child.dep_ in {'advmod', 'amod', 'npadvmod'} and\
    #                  child.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and child.lemma_ not in VAGUE_ADVMODS]
    # parts.extend(adverbs_after)

    # Get adverbial modifiers after the verb
    for child in verb.rights:
        if (
            child.dep_ in {"advmod", "amod", "npadvmod"}
            and child.pos_ not in {"SCONJ", "CCONJ", "PART", "DET"}
            and child.lemma_ not in _VAGUE_ADVMODS
        ):
            # Include the noun and its modifiers
            noun_phrase, prep_phrases = get_compound_parts(child)
            parts.append(noun_phrase)

    # Get auxiliaries and negations after the verb
    aux = [
        child.lemma_
        for child in verb.rights
        if child.dep_ in {"aux", "auxpass", "neg"} and child.lemma_ not in _VAGUE_AUX
    ]
    parts.extend(aux)

    for child in verb.children:
        if child.dep_ in {"xcomp"}:
            # Include the xcomp verb lemma directly
            parts.append(child.lemma_)
            # Optionally include auxiliaries and modifiers of the xcomp verb
            child_verb_phrase = get_verb_phrase(child)
            parts.extend(child_verb_phrase[1:])

    return [" ".join(parts)]


def get_verb_subjects(verb):
    """
    Retrieve the subjects of a given verb, including through coordination and inheritance.
    Coordination refers to when multiple subjects are linked by a conjunction (e.g., "and" or "or") and share the same verb.
    Inheritance occurs when a subject is implied from a previous clause or sentence and applies to a following verb without being explicitly repeated.

    For compound nouns, each component is included separately.

    PASSIVE VOICE HANDLING (fix Punto 13):
      - Passivo con agente ("X was Vd by Y"):
          Y (→ pobj dell'agent) è il soggetto SEMANTICO → va in Who
          X (nsubjpass) è il paziente → NON va in Who (andrà in What via get_verb_objects)
      - Passivo senza agente ("X was Vd"):
          Nessun agente identificabile; X (nsubjpass) rimane in Who
          come miglior approssimazione disponibile.
      - Passivo coordinato ("X was written and reviewed by Y"):
          L'agent può essere figlio del verbo conj, non del ROOT.
          Controlliamo anche i figli dei conj di tipo VERB.

    Args:
        verb: The verb token.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main subject and its prepositional phrases.

    """
    subjects = []

    # Handle acl and relcl dependencies
    if verb.dep_ in {"acl", "relcl"}:
        # If the verb has its own subject, use it
        nsubjs = [child for child in verb.children if child.dep_ == "nsubj"]
        if nsubjs:
            for subj in nsubjs:
                if subj.lower_ in {"who", "which", "that", "whom", "whose"}:
                    # Use the modified noun as the subject
                    modified_noun = verb.head
                    main_part, prep_parts = get_compound_parts(
                        modified_noun, lemmatize=True
                    )
                    subjects.append((main_part, prep_parts))
                else:
                    subjects.extend(extract_subjects(subj))
        else:
            # Use the modified noun as the subject
            modified_noun = verb.head
            main_part, prep_parts = get_compound_parts(modified_noun, lemmatize=True)
            subjects.append((main_part, prep_parts))
        return subjects

    # ── PASSIVE VOICE HANDLING ────────────────────────────────────────────────
    # Determine if this is a passive construction (nsubjpass or auxpass present)
    _nsubjpass_labels = {"nsubjpass", "nsubj:pass"}
    _auxpass_labels   = {"auxpass",   "aux:pass"}
    children_deps = {child.dep_ for child in verb.children}
    is_passive = bool(
        (children_deps & _nsubjpass_labels) or
        (children_deps & _auxpass_labels)
    )

    if is_passive:
        # Look for agent: first in direct children, then in conj verbs
        agent_pobj = None

        for child in verb.children:
            if child.dep_ == "agent":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        agent_pobj = gc
                        break
            if agent_pobj:
                break

        # Coordinated passive: agent may attach to a conj verb
        if agent_pobj is None:
            for child in verb.children:
                if child.dep_ == "conj" and child.pos_ == "VERB":
                    for gc in child.children:
                        if gc.dep_ == "agent":
                            for ggc in gc.children:
                                if ggc.dep_ == "pobj":
                                    agent_pobj = ggc
                                    break
                        if agent_pobj:
                            break
                if agent_pobj:
                    break

        if agent_pobj is not None:
            # Passive WITH agent: agent → Who; nsubjpass → What (handled by get_verb_objects)
            subjects.extend(extract_subjects(agent_pobj))
            return subjects
        else:
            # Passive WITHOUT agent: nsubjpass stays in Who (best available approximation)
            nsubjpass_tokens = [
                child for child in verb.children
                if child.dep_ in _nsubjpass_labels
            ]
            for subj in nsubjpass_tokens:
                subjects.extend(extract_subjects(subj))
            if subjects:
                return subjects
    # ── END PASSIVE VOICE HANDLING ────────────────────────────────────────────

    # Standard active voice: use nsubj
    direct_subjects = [
        child for child in verb.children if child.dep_ in {"nsubj"}
    ]

    for subj in direct_subjects:
        subjects.extend(extract_subjects(subj))

    # Handle clausal subjects (csubj)
    csubj_subjects = [child for child in verb.children if child.dep_ == "csubj"]
    for csubj in csubj_subjects:
        clause = extract_clause(csubj)
        subjects.append((clause, []))  # No preps in clause

    # Inherit subjects from ancestor verbs if none found
    if not subjects:
        for ancestor in verb.ancestors:
            ancestor_subjects = [
                child
                for child in ancestor.children
                if child.dep_ in {"nsubj", "nsubjpass", "nsubj:pass", "csubj"}
            ]
            for anc_subj in ancestor_subjects:
                subjects.extend(extract_subjects(anc_subj))
                if subjects:
                    break
            if subjects:
                break

    # Inherit subjects from head verb if verb is a conjunct
    if not subjects and verb.dep_ == "conj" and verb.head.pos_ == "VERB":
        subjects = get_verb_subjects(verb.head)

    # Handle 'conj' nouns connected to the verb
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in {"NOUN", "PROPN", "PRON"}:
            subjects.extend(extract_subjects(child))

    # Handle imperative sentences (assuming 'you' as subject)
    if not subjects and verb.head == verb:
        # Check if the verb is in imperative mood
        if verb.morph.get("Mood") == ["Imp"] or verb.tag_ == "VB":
            subjects.append(("you", []))
    # Remove duplicates while preserving order

    return subjects


def extract_subjects(subject_token):
    """
    Extract the subject and its compound components.
    Args:
        subject_token (spacy.tokens.Token): The subject token.
    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main subject and its prepositional phrases.
    """
    subjects = []

    if subject_token.pos_ in {"NOUN", "PROPN", "PRON"}:
        # Extract compounds and the main noun
        main_part, prep_parts = get_compound_parts(subject_token, lemmatize=True)
        subjects.append((main_part, prep_parts))

    # Additionally, handle conjunct subjects (e.g., "Alice and Bob")
    conjunct_subjects = [
        conj
        for conj in subject_token.conjuncts
        if conj.pos_ in {"NOUN", "PROPN", "PRON"}
    ]
    for conj in conjunct_subjects:
        main_part, prep_parts = get_compound_parts(conj, lemmatize=True)
        subjects.append((main_part, prep_parts))

    return subjects


def get_compound_parts(token, lemmatize=True):
    """
    Retrieve the main token and its compound modifiers as a single string.
    Also includes adverbial and adjectival modifiers of adjectives.

    Args:
        token (spacy.tokens.Token): The token to extract compounds from.
        lemmatize (bool, optional): Whether to lemmatize the extracted parts.

    Returns:
        Tuple[str, List[str]]: A tuple containing the compound phrase and a list of prepositional phrases.
    """
    parts = []
    prep_parts = []

    # Extract determiners (excluding 'the', 'a', 'an')
    dets = [
        child
        for child in token.children
        if (child.dep_ == "det") and child.lemma_ not in {"the", "a", "an"}
    ]
    if lemmatize:
        parts.extend([child.lemma_ for child in dets])
    else:
        parts.extend([child.text for child in dets])

    # Extract modifiers
    # Adjusted condition in get_compound_parts
    modifiers = []
    for child in token.children:
        if (
            child.dep_ in {"compound", "amod", "nmod", "poss", "case"}
            or (
                child.dep_ in {"advmod", "npadvmod"}
                and token.pos_ in {"VERB", "AUX", "ADJ", "NOUN", "PROPN"}
            )
        ) and child.pos_ not in {"SCONJ", "CCONJ", "PART"}:
            modifiers.append(child)
            # Include modifiers of modifiers, including advmod for adjectives and nouns
            modifiers.extend(
                [
                    gc
                    for gc in child.children
                    if gc.dep_ in {"compound", "amod", "nmod", "poss", "case", "advmod"}
                ]
            )

    # Sort modifiers based on their position in the text
    modifiers = sorted(set(modifiers), key=lambda x: x.i)

    # Add all modifier text
    for modifier in modifiers:
        if lemmatize:
            parts.append(modifier.lemma_.lower())
        else:
            parts.append(modifier.text)

    # Add the main token's text
    if lemmatize:
        parts.append(token.lemma_.lower())
    else:
        parts.append(token.text)

    # Handle prepositional phrases separately
    preps = [child for child in token.children if child.dep_ == "prep"]
    for prep in preps:
        # Process the preposition and its conjuncts
        prep_conjuncts = [prep] + list(prep.conjuncts)
        for p in prep_conjuncts:
            pobj_list = [child for child in p.children if child.dep_ == "pobj"]
            for pobj in pobj_list:
                # Process the pobj and its conjuncts
                pobj_conjuncts = [pobj] + list(pobj.conjuncts)
                for pobj_item in pobj_conjuncts:
                    # For each pobj_conjunct, create a separate prep_phrase
                    prep_phrase = [p.lemma_]  # Start with the preposition
                    pobj_main, pobj_preps = get_compound_parts(
                        pobj_item, lemmatize=lemmatize
                    )
                    prep_phrase.append(pobj_main)
                    # Handle nested prepositional phrases
                    for nested_prep in pobj_preps:
                        prep_phrase.append(nested_prep)
                    # Join and append to prep_parts
                    prep_parts.append(" ".join(prep_phrase))

    main_part = " ".join(parts)

    return main_part, prep_parts


def get_conjuncts(token, visited=None):
    """
    Recursively retrieve the token and its conjuncts, avoiding infinite loops.

    Args:
        token (spacy.tokens.Token): The token to find conjuncts for.
        visited (Set[spacy.tokens.Token], optional): Set of already visited tokens.

    Returns:
        List[Token]: A list of tokens including the original token and its conjuncts.
    """
    if visited is None:
        visited = set()
    conjuncts = [token]
    visited.add(token)
    for conj in token.conjuncts:
        if conj not in visited:
            conjuncts.extend(get_conjuncts(conj, visited))
    return conjuncts


def extract_clause(token):
    """
    Extract the full clause connected to a token.
    """
    clause_tokens = list(token.subtree)
    clause_tokens = sorted(clause_tokens, key=lambda x: x.i)
    lemmas = [t.lemma_ for t in clause_tokens]
    clause = " ".join(lemmas)
    return clause


def get_verb_objects(verb):
    """
    Retrieve the objects of a given verb, considering passive voice and the presence of agents.
    This function handles various cases to accurately extract objects, including:
    - Direct objects
    - Indirect objects
    - Objects in prepositional phrases
    - Passive constructions with or without agents
    - Clausal complements
    - Objects from xcomp dependencies
    - Inheritance from conjoined verbs

    Args:
        verb (spacy.tokens.Token): The verb token.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main object and its prepositional phrases.
    """
    objects = []

    # Helper function to process and extend objects list, including conjuncts
    def process_objects(object_tokens):
        for obj in object_tokens:
            objects.extend(extract_objects(obj))

    # Check if the verb has an agent (passive voice with agent)
    agents = [child for child in verb.children if child.dep_ == "agent"]
    if agents:
        # If there's an agent, treat the nsubjpass as the object
        # Support both legacy (nsubjpass) and UD (nsubj:pass) labels
        nsubjpass = [child for child in verb.children if child.dep_ in {"nsubjpass", "nsubj:pass"}]
        process_objects(nsubjpass)
    else:
        # Proceed with standard object extraction
        # 1. Direct Objects (e.g., "eat an apple")
        direct_objects = [
            child
            for child in verb.children
            if child.dep_ in {"dobj", "attr", "oprd", "acomp"}
        ]
        process_objects(direct_objects)

        # 2. Indirect Objects (e.g., "give me the book")
        indirect_objects = [
            child for child in verb.children if child.dep_ in {"iobj", "dative"}
        ]
        process_objects(indirect_objects)

        # 3. Objects in Prepositional Phrases (e.g., "look at the sky")
        prep_phrases = [child for child in verb.children if child.dep_ == "prep"]
        for prep in prep_phrases:
            # Handles Conjuncts in Prepositional Phrases
            preps = get_conjuncts(prep)
            for p in preps:
                pobj = [child for child in p.children if child.dep_ == "pobj"]
                process_objects(pobj)

        # 4. If no direct objects, include prepositional phrases as objects
        if not objects:
            for prep in prep_phrases:
                prep_text = prep.lemma_
                pobj = [child for child in prep.children if child.dep_ == "pobj"]
                for obj in pobj:
                    main_part, prep_parts = get_compound_parts(obj)
                    full_object = prep_text + " " + main_part
                    objects.append((full_object, prep_parts))

    # 5. Objects in Noun Phrase Adverbial Modifiers (e.g., "He arrived yesterday")
    npadvmod_objects = [
        child
        for child in verb.children
        if (child.dep_ == "npadvmod") and (child.pos_ in {"PROPN", "PRON", "NOUN"})
    ]
    process_objects(npadvmod_objects)

    # 6. Handle clausal complements (ccomp)
    for child in verb.children:
        if child.dep_ == "ccomp":
            # Include the objects of the ccomp verb
            ccomp_objects = get_verb_objects(child)
            objects.extend(ccomp_objects)

    # 7. Handle xcomp dependencies: include objects from xcomp verb
    xcomp_children = [child for child in verb.children if child.dep_ == "xcomp"]
    for xcomp_child in xcomp_children:
        xcomp_objects = get_verb_objects(xcomp_child)
        if xcomp_objects:
            objects.extend(xcomp_objects)
        else:
            # If no objects found, try to get direct objects from the xcomp verb
            xcomp_dobjs = [
                child
                for child in xcomp_child.children
                if child.dep_ in {"dobj", "attr", "oprd"}
            ]
            process_objects(xcomp_dobjs)

    # 8. Inherit objects from conjoined verbs if none found
    if not objects:
        for child in verb.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                conj_objects = get_verb_objects(child)
                if conj_objects:
                    objects.extend(conj_objects)
                    break  # Avoid duplicates by stopping after the first found

    return objects


def extract_objects(object_token):
    """
    Extract the object and its compound components, including prepositional phrases.

    Args:
        object_token (spacy.tokens.Token): The object token.
    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main object and its prepositional phrases.
    """
    objects = []

    # Check if the token is a noun, proper noun, or pronoun
    if (object_token.pos_ in {"NOUN", "PROPN", "PRON"}) or (
        object_token.dep_ in ("acomp")
    ):
        # Extract compounds and the main noun
        main_part, prep_parts = get_compound_parts(object_token)
        # If the object is a 'pobj', include the preposition
        if object_token.dep_ == "pobj":
            preposition = object_token.head.lemma_
            main_part = preposition + " " + main_part
        objects.append((main_part, prep_parts))

    # Additionally, handle conjunct objects (e.g., "apples and oranges")

    conjunct_objects = [
        conj
        for conj in object_token.conjuncts
        if conj.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}
    ]
    for conj in conjunct_objects:
        # Extract compounds and the main noun for the conjunct
        main_part, prep_parts = get_compound_parts(conj)
        # If the conjunct object is a 'pobj', include the preposition
        if conj.dep_ == "pobj":
            preposition = conj.head.lemma_
            main_part = preposition + " " + main_part
        objects.append((main_part, prep_parts))

    return objects
