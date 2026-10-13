import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.app.core.exceptions import InvalidReferenceTextError

# Extended ARPAbet lexicon for common interview and delivery practice sentences
_CMU_SUBSET: Dict[str, List[str]] = {
    # Core function & conversational words
    "hello": ["HH", "AH0", "L", "OW1"],
    "world": ["W", "ER1", "L", "D"],
    "this": ["DH", "IH1", "S"],
    "that": ["DH", "AE1", "T"],
    "is": ["IH1", "Z"],
    "a": ["AH0"],
    "an": ["AE1", "N"],
    "the": ["DH", "AH0"],
    "test": ["T", "EH1", "S", "T"],
    "i": ["AY1"],
    "my": ["M", "AY1"],
    "me": ["M", "IY1"],
    "we": ["W", "IY1"],
    "our": ["AW1", "ER0"],
    "you": ["Y", "UW1"],
    "your": ["Y", "AO1", "R"],
    "they": ["DH", "EY1"],
    "it": ["IH1", "T"],
    "have": ["HH", "AE1", "V"],
    "has": ["HH", "AE1", "Z"],
    "had": ["HH", "AE1", "D"],
    "been": ["B", "IH1", "N"],
    "am": ["AE1", "M"],
    "are": ["AA1", "R"],
    "was": ["W", "AA1", "Z"],
    "were": ["W", "ER1"],
    "will": ["W", "IH1", "L"],
    "would": ["W", "UH1", "D"],
    "can": ["K", "AE1", "N"],
    "could": ["K", "UH1", "D"],
    "working": ["W", "ER1", "K", "IH0", "NG"],
    "work": ["W", "ER1", "K"],
    "on": ["AA1", "N"],
    "in": ["IH1", "N"],
    "at": ["AE1", "T"],
    "to": ["T", "UW1"],
    "for": ["F", "AO1", "R"],
    "with": ["W", "IH1", "DH"],
    "about": ["AH0", "B", "AW1", "T"],
    "like": ["L", "AY1", "K"],
    "three": ["TH", "R", "IY1"],
    "two": ["T", "UW1"],
    "one": ["W", "AH1", "N"],
    "four": ["F", "AO1", "R"],
    "five": ["F", "AY1", "V"],
    "years": ["Y", "IH1", "R", "Z"],
    "year": ["Y", "IH1", "R"],
    "building": ["B", "IH1", "L", "D", "IH0", "NG"],
    "build": ["B", "IH1", "L", "D"],
    "speech": ["S", "P", "IY1", "CH"],
    "and": ["AE1", "N", "D"],
    "or": ["AO1", "R"],
    "but": ["B", "AH1", "T"],
    "because": ["B", "IH0", "K", "AO1", "Z"],
    "interview": ["IH1", "N", "T", "ER0", "V", "Y", "UW2"],
    "delivery": ["D", "IH0", "L", "IH1", "V", "ER0", "IY0"],
    "analysis": ["AH0", "N", "AE1", "L", "AH0", "S", "AH0", "S"],
    "system": ["S", "IH1", "S", "T", "AH0", "M"],
    "systems": ["S", "IH1", "S", "T", "AH0", "M", "Z"],
    "distributed": ["D", "IH0", "S", "T", "R", "IH1", "B", "Y", "AH0", "T", "IH0", "D"],
    "um": ["AH1", "M"],
    "uh": ["AH1"],
    "er": ["ER1"],
    "know": ["N", "OW1"],
    # Interview practice sentences vocabulary
    "greatest": ["G", "R", "EY1", "T", "AH0", "S", "T"],
    "strength": ["S", "T", "R", "EH1", "NG", "K", "TH"],
    "strengths": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
    "weakness": ["W", "IY1", "K", "N", "AH0", "S"],
    "communication": ["K", "AH0", "M", "Y", "UW2", "N", "AH0", "K", "EY1", "SH", "AH0", "N"],
    "problem": ["P", "R", "AA1", "B", "L", "AH0", "M"],
    "problems": ["P", "R", "AA1", "B", "L", "AH0", "M", "Z"],
    "solving": ["S", "AA1", "L", "V", "IH0", "NG"],
    "solve": ["S", "AA1", "L", "V"],
    "experience": ["IH0", "K", "S", "P", "IH1", "R", "IY0", "AH0", "N", "S"],
    "team": ["T", "IY1", "M"],
    "teams": ["T", "IY1", "M", "Z"],
    "project": ["P", "R", "AA1", "JH", "EH0", "K", "T"],
    "projects": ["P", "R", "AA1", "JH", "EH0", "K", "T", "S"],
    "software": ["S", "AO1", "F", "T", "W", "EH2", "R"],
    "engineer": ["EH1", "N", "JH", "AH0", "N", "IH1", "R"],
    "engineering": ["EH2", "N", "JH", "AH0", "N", "IH1", "R", "IH0", "NG"],
    "developer": ["D", "IH0", "V", "EH1", "L", "AH0", "P", "ER0"],
    "development": ["D", "IH0", "V", "EH1", "L", "AH0", "P", "M", "AH0", "N", "T"],
    "technology": ["T", "EH0", "K", "N", "AA1", "L", "AH0", "JH", "IY0"],
    "leader": ["L", "IY1", "D", "ER0"],
    "leadership": ["L", "IY1", "D", "ER0", "SH", "IH2", "P"],
    "challenge": ["CH", "AE1", "L", "AH0", "N", "JH"],
    "challenges": ["CH", "AE1", "L", "AH0", "N", "JH", "IH0", "Z"],
    "opportunity": ["AA2", "P", "ER0", "T", "UW1", "N", "AH0", "T", "IY0"],
    "excited": ["IH0", "K", "S", "AY1", "T", "IH0", "D"],
    "collaborate": ["K", "AH0", "L", "AE1", "B", "ER0", "EY2", "T"],
    "collaboration": ["K", "AH0", "L", "AE2", "B", "ER0", "EY1", "SH", "AH0", "N"],
    "learn": ["L", "ER1", "N"],
    "learning": ["L", "ER1", "N", "IH0", "NG"],
    "goal": ["G", "OW1", "L"],
    "goals": ["G", "OW1", "L", "Z"],
    "active": ["AE1", "K", "T", "IH0", "V"],
    "listening": ["L", "IH1", "S", "AH0", "N", "IH0", "NG"],
    "clear": ["K", "L", "IH1", "R"],
    "clearly": ["K", "L", "IH1", "R", "L", "IY0"],
    "think": ["TH", "IH1", "NG", "K"],
    "thank": ["TH", "AE1", "NG", "K"],
    "thanks": ["TH", "AE1", "NG", "K", "S"],
    "time": ["T", "AY1", "M"],
    "role": ["R", "OW1", "L"],
    "company": ["K", "AH1", "M", "P", "AH0", "N", "IY0"],
    "value": ["V", "AE1", "L", "Y", "UW0"],
    "values": ["V", "AE1", "L", "Y", "UW0", "Z"],
    "today": ["T", "AH0", "D", "EY1"],
    "name": ["N", "EY1", "M"],
    "alex": ["AE1", "L", "IH0", "K", "S"],
    "previous": ["P", "R", "IY1", "V", "IY0", "AH0", "S"],
    "lead": ["L", "IY1", "D"],
    "led": ["L", "EH1", "D"],
    "improve": ["IH0", "M", "P", "R", "UW1", "V"],
    "improvement": ["IH0", "M", "P", "R", "UW1", "V", "M", "AH0", "N", "T"],
    "reliable": ["R", "IH0", "L", "AY1", "AH0", "B", "AH0", "L"],
    "reliability": ["R", "IY0", "L", "AY2", "AH0", "B", "IH1", "L", "AH0", "T", "IY0"],
    "skill": ["S", "K", "IH1", "L"],
    "skills": ["S", "K", "IH1", "L", "Z"],
    "data": ["D", "EY1", "T", "AH0"],
    "code": ["K", "OW1", "D"],
    "design": ["D", "IH0", "Z", "AY1", "N"],
    "manage": ["M", "AE1", "N", "AH0", "JH"],
    "management": ["M", "AE1", "N", "AH0", "JH", "M", "AH0", "N", "T"],
    "product": ["P", "R", "AA1", "D", "AH0", "K", "T"],
    "products": ["P", "R", "AA1", "D", "AH0", "K", "T", "S"],
    "service": ["S", "ER1", "V", "AH0", "S"],
    "services": ["S", "ER1", "V", "AH0", "S", "AH0", "Z"],
    "customer": ["K", "AH1", "S", "T", "AH0", "M", "ER0"],
    "customers": ["K", "AH1", "S", "T", "AH0", "M", "ER0", "Z"],
    "technical": ["T", "EH1", "K", "N", "IH0", "K", "AH0", "L"],
    "solution": ["S", "AH0", "L", "UW1", "SH", "AH0", "N"],
    "solutions": ["S", "AH0", "L", "UW1", "SH", "AH0", "N", "Z"],
    "good": ["G", "UH1", "D"],
    "great": ["G", "R", "EY1", "T"],
    "well": ["W", "EH1", "L"],
    "speak": ["S", "P", "IY1", "K"],
    "speaking": ["S", "P", "IY1", "K", "IH0", "NG"],
    "talk": ["T", "AO1", "K"],
    "question": ["K", "W", "EH1", "S", "CH", "AH0", "N"],
    "questions": ["K", "W", "EH1", "S", "CH", "AH0", "N", "Z"],
    "answer": ["AE1", "N", "S", "ER0"],
    "answers": ["AE1", "N", "S", "ER0", "Z"],
    "sentence": ["S", "EH1", "N", "T", "AH0", "N", "S"],
    "worked": ["W", "ER1", "K", "T"],
    "she": ["SH", "IY1"],
    "sells": ["S", "EH1", "L", "Z"],
    "sell": ["S", "EH1", "L"],
    "seashells": ["S", "IY1", "SH", "EH1", "L", "Z"],
    "sea": ["S", "IY1"],
    "shell": ["SH", "EH1", "L"],
    "shells": ["SH", "EH1", "L", "Z"],
    "by": ["B", "AY1"],
    "seashore": ["S", "IY1", "SH", "AO1", "R"],
    "shore": ["SH", "AO1", "R"],
    "sure": ["SH", "UH1", "R"],
    "did": ["D", "IH1", "D"],
    "done": ["D", "AH1", "N"],
    "doing": ["D", "UW1", "IH0", "NG"],
    "managed": ["M", "AE1", "N", "AH0", "JH", "D"],
    "designed": ["D", "IH0", "Z", "AY1", "N", "D"],
    "solved": ["S", "AA1", "L", "V", "D"],
    "built": ["B", "IH1", "L", "T"],
    "created": ["K", "R", "IY0", "EY1", "T", "IH0", "D"],
    "create": ["K", "R", "IY0", "EY1", "T"],
    "joined": ["JH", "OY1", "N", "D"],
    "join": ["JH", "OY1", "N"],
    "helped": ["HH", "EH1", "L", "P", "T"],
    "help": ["HH", "EH1", "L", "P"],
    "improved": ["IH0", "M", "P", "R", "UW1", "V", "D"],
    "tested": ["T", "EH1", "S", "T", "IH0", "D"],
    "delivered": ["D", "IH0", "L", "IH1", "V", "ER0", "D"],
    "deliver": ["D", "IH0", "L", "IH1", "V", "ER0"],
    "how": ["HH", "AW1"],
    "much": ["M", "AH1", "CH"],
    "wood": ["W", "UH1", "D"],
    "should": ["SH", "UH1", "D"],
    "chuck": ["CH", "AH1", "K"],
}

# Rule-based fallback mapping single letters to approximate canonical ARPAbet sounds
_FALLBACK_LETTER_MAP: Dict[str, str] = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH",
    "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "AH", "v": "V", "w": "W", "x": "K", "y": "Y",
    "z": "Z",
}

# Standard ARPAbet-39 base phonemes to eSpeak / IPA phonetic tokens
# Used by Wav2Vec2 phoneme CTC models (e.g. facebook/wav2vec2-lv-60-espeak-cv-ft)
ARPABET_TO_ESPEAK: Dict[str, str] = {
    # Vowels & Diphthongs
    "AA": "ɑː",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔː",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɜː",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "iː",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "uː",
    # Consonants
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


@dataclass
class PhonemeReference:
    phoneme_arpabet: str       # e.g. "AH0"
    phoneme_base: str          # e.g. "AH"
    model_token: Optional[str] # e.g. "ʌ"
    is_supported: bool         # True if mapped to model token


@dataclass
class WordReference:
    word: str
    phonemes: List[PhonemeReference]


def strip_stress(phoneme: str) -> str:
    """Removes numeric stress annotations from ARPAbet phoneme (e.g. 'AH0' -> 'AH')."""
    return re.sub(r"\d+", "", phoneme).upper()


def word_to_phonemes(word: str) -> List[str]:
    """
    Converts an English word to canonical ARPAbet phoneme sequence.
    Uses dictionary lookup with rule-based phonetic fallback for OOV words.
    """
    clean_word = re.sub(r"[^\w]", "", word).lower()
    if not clean_word:
        return []

    if clean_word in _CMU_SUBSET:
        return list(_CMU_SUBSET[clean_word])

    # Rule-based fallback for OOV words
    phonemes: List[str] = []
    i = 0
    n = len(clean_word)
    while i < n:
        if i < n - 1:
            pair = clean_word[i : i + 2]
            if pair in ("th",):
                phonemes.append("TH")
                i += 2
                continue
            elif pair in ("sh",):
                phonemes.append("SH")
                i += 2
                continue
            elif pair in ("ch",):
                phonemes.append("CH")
                i += 2
                continue
            elif pair in ("ee", "ea"):
                phonemes.append("IY")
                i += 2
                continue
            elif pair in ("oo",):
                phonemes.append("UW")
                i += 2
                continue
            elif pair in ("ai", "ay"):
                phonemes.append("EY")
                i += 2
                continue
            elif pair in ("oi", "oy"):
                phonemes.append("OY")
                i += 2
                continue
            elif pair in ("ou", "ow"):
                phonemes.append("AW")
                i += 2
                continue
            elif pair in ("ng",):
                phonemes.append("NG")
                i += 2
                continue
            elif pair[0] == pair[1] and pair[0] in _FALLBACK_LETTER_MAP:
                # Collapse double consonant letters in spelling (e.g. 'll' -> 'L', 'ss' -> 'S')
                phonemes.append(_FALLBACK_LETTER_MAP[pair[0]])
                i += 2
                continue

        # Terminal silent 'e' guard: word length >= 3 and ends in consonant + 'e'
        if i == n - 1 and clean_word[i] == "e" and n >= 3 and clean_word[i - 1] not in ("e", "a", "i", "o", "u"):
            i += 1
            continue

        # Terminal or vowel-like 'y'
        if clean_word[i] == "y":
            if i == n - 1 or n == 1 or clean_word[i - 1] not in ("a", "e", "i", "o", "u"):
                phonemes.append("AY")
                i += 1
                continue

        char = clean_word[i]
        if char in _FALLBACK_LETTER_MAP:
            phonemes.append(_FALLBACK_LETTER_MAP[char])
        i += 1

    return phonemes or ["AH"]


def validate_reference_text(text: Optional[str]) -> str:
    """
    Validates and normalizes reference text for pronunciation assessment.
    Raises InvalidReferenceTextError for empty, excessively long, or unsupported text.
    """
    if text is None:
        raise InvalidReferenceTextError("Reference text cannot be null.")

    normalized = " ".join(text.strip().split())
    if not normalized:
        raise InvalidReferenceTextError("Reference text cannot be empty or whitespace only.")

    words = normalized.split()
    if len(words) > 80:
        raise InvalidReferenceTextError(
            f"Reference text exceeds maximum supported practice length of 80 words ({len(words)} words provided)."
        )

    if len(normalized) > 500:
        raise InvalidReferenceTextError(
            f"Reference text exceeds maximum allowed length of 500 characters ({len(normalized)} chars provided)."
        )

    # Check for excessive unprintable or non-Latin/control characters
    sanitized = re.sub(r"[^\w\s.,!?'\"-]", "", normalized)
    if len(sanitized) < len(normalized) * 0.7:
        raise InvalidReferenceTextError(
            "Reference text contains too many unsupported symbols or non-English characters."
        )

    return normalized


def text_to_reference_phonemes(text: str) -> List[WordReference]:
    """
    Parses a reference sentence into an ordered sequence of WordReference objects
    with explicit phoneme mapping and support flags.
    """
    norm_text = validate_reference_text(text)
    words = [re.sub(r"[^\w']", "", w).strip() for w in norm_text.split()]
    words = [w for w in words if w]

    word_refs: List[WordReference] = []

    for word in words:
        raw_phonemes = word_to_phonemes(word)
        ph_refs: List[PhonemeReference] = []

        for p in raw_phonemes:
            base_p = strip_stress(p)
            model_tok = ARPABET_TO_ESPEAK.get(base_p, None)
            is_sup = model_tok is not None

            ph_refs.append(
                PhonemeReference(
                    phoneme_arpabet=p,
                    phoneme_base=base_p,
                    model_token=model_tok,
                    is_supported=is_sup,
                )
            )

        word_refs.append(WordReference(word=word, phonemes=ph_refs))

    return word_refs
