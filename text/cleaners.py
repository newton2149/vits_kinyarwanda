""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize
from .numbers import normalize_numbers
from cvutils import Phonemiser
from gruut import sentences


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True, preserve_punctuation=True, with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def bengali_cleaners(text):
    """Pipeline for Bengali text, including lowercase + punctuation + stress"""
    # text = lowercase(text)
    text = phonemize(text, backend="espeak", language="bn", strip=True, preserve_punctuation=True, with_stress=True, njobs=8)
    # text = collapse_whitespace(text)
    return text

def kinyarwanda_cleaner(text):
    phonemes= []
    p = Phonemiser('rw')
    text_split = text.split(" ")
    for t in text_split:
        phonemes.append(p.phonemise(t))
    phonemes = " ".join(phonemes)
    # print(f'Text: {text} ---> Phoneme: {phonemes} ',end='\r')
    return phonemes

def french_cleaner(text):


    phoneme_sentence = []

    for sent in sentences(text, lang="fr-fr"):
        for word in sent:
            if word.phonemes:
                phoneme_sentence.extend(word.phonemes)
            phoneme_sentence.extend(" ")


    phoneme = "".join(phoneme_sentence).strip()
    phoneme = phoneme.replace("|","*")