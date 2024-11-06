import spacy
import warnings
from whodidwhat.resources import _valences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



_nlp_spacy = None
_nlp_stanza = None



def spacynlp(text):
    """
    Process the text using the spaCy NLP pipeline.
    """
    nlp = get_spacy_nlp()

    return nlp(text)

def ensure_wordnet_downloaded():
    import nltk
    from nltk.data import find
    try:
        find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')



def get_spacy_nlp():
    global _nlp_spacy
    if _nlp_spacy is None:
        ensure_wordnet_downloaded()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_spacy = spacy.load('en_core_web_trf')
    return _nlp_spacy

def get_stanza_nlp():
    import stanza
    global _nlp_stanza
    if _nlp_stanza is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,coref', verbose=False)
    return _nlp_stanza



def compute_valence(text):
    """
    Compute the valence of a given text based on positive and negative words using Vader Sentiment Analysis.

    Args:
        text (str): The text to analyze.

    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    analyzer = SentimentIntensityAnalyzer()

    vs = analyzer.polarity_scores(text)

    if vs['compound'] > 0.65:
        return 'positive'
    elif vs['compound'] < -0.65:
        return 'negative'
    else:
        return 'neutral'