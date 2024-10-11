
import spacy
import subprocess


def _valences(language="english"):

    if language == "english":
        from valence.english import _positive, _negative, _ambivalent


    elif language == "italian":
        from valence.italian import _positive, _negative, _ambivalent

    return _positive, _negative, _ambivalent


_VAGUE_ADVMODS = {'only', 'respectively', 'actually', 'basically', 'generally', 'normally', 'usually', 'approximately', 'roughly', 'virtually', 
                 'exactly', 'precisely', 'literally', 'effectively', 'essentially','more'}

_VAGUE_AUX = {'be','have','to', 'shall', 'would', 'should','is'}

_VAGUE_ADJ = {
    "certain", "significant", "some", "various", "several", "numerous", "other", "different", "specific", "typical", "important",
    "general", "usual", "frequent", "considerable", "sufficient", "major", "relevant", "adequate", "appropriate", "minor",
    "potential", "many", "few", "particular", "additional", "existing", "regular", "normal", "ordinary", "standard", "common",
    "average", "basic", "random", "miscellaneous", "diverse", "assorted", "respective", "relative", "suitable", "proper", "given",
    "multiple", "sundry", "arbitrary", "undefined", "unspecified", "approximate", "estimated", "nominal", "generic", "universal",
    "conventional", "customary", "moderate", "reasonable", "notable", "substantial", "possible", "probable", "likely", "unlikely",
    "primary", "secondary", "tertiary", "minimal", "maximal", "optimal", "various", "countless", "innumerable", "limited",
    "extensive", "abundant", "scarce", "ample", "sparse", "much", "current"
    }
