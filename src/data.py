"""Data loading and deterministic answer scoring.

Two splits:

* **TRAIN** — `deepeyes/data_0.1.2_visual_toolbox_v2.parquet`, 22,362 rows. Read with
  pyarrow only (no pandas, no `datasets`). See GOAL S18 for the schema.
* **EVAL** — `vstar/{direct_attributes,relative_position}/sa_*.{jpg,json}`, 191 items.

This module replaces the DeepEyes LLM judge with a deterministic checker (Gate 0.5,
GOAL S9). It imports on the Mac and on the rig: no torch, no transformers, no vLLM.

Frozen interface (the trainer imports these five names):

    load_train(limit=None, seed=0, filtered_only=True) -> list[Sample]
    load_vstar(limit=None)                             -> list[Sample]
    answer_is_scorable(gold)                           -> bool
    normalize_answer(s)                                -> str
    answer_correct(pred, gold, options=None)           -> bool

Two facts a caller must know; both are measured, not assumed (see `.notes/`):

1. **V*Bench `options[0]` is ALWAYS the gold** — 191/191, verified against the
   `label` field of `test_questions.jsonl`. `load_vstar` therefore SHUFFLES the
   options per sample and records the post-shuffle position in
   `meta["gold_index"]`. Never present `options` to a model unshuffled.
2. **Train golds are always full sentences.** Not one bare word in 22,362 rows.
   "Yes, the color of the pole is gray.", never "gray". Scoring must reduce the
   sentence, which is what `answer_correct` does.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import string
from pathlib import Path

from src.contract import Sample

# --- paths --------------------------------------------------------------------

#: Root of the data tree. `/srv/ai/data` on the rig; override on the Mac.
DATA_ROOT = Path(os.environ.get("COST_AWARE_DATA_ROOT", "/srv/ai/data"))

TRAIN_PARQUET = DATA_ROOT / "deepeyes" / "data_0.1.2_visual_toolbox_v2.parquet"
VSTAR_ROOT = DATA_ROOT / "vstar"

#: The V*Bench subdirs we evaluate on. `GPT4V-hard/` and `OCR/` are excluded on
#: purpose (GOAL S10) and their jpg/json counts do not even match.
VSTAR_SUBDIRS = ("direct_attributes", "relative_position")


def _require(path: Path, what: str) -> Path:
    """Fail with a readable message instead of an obscure IOError deep in a loader."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Set COST_AWARE_DATA_ROOT to the directory "
            f"that holds deepeyes/ and vstar/ (it is /srv/ai/data on the rig). "
            f"Current COST_AWARE_DATA_ROOT={DATA_ROOT}"
        )
    return path


# --- normalization ------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({c: " " for c in string.punctuation})

#: Leading tokens that assert / deny. Only ever read at position 0, or as the whole
#: string. That is what keeps "not sure" from parsing as "no" -- "not" is not here,
#: and "no" only counts when it is the literal first token.
_YES_TOKENS = frozenset({"yes", "yeah", "yep", "yup", "true", "correct", "affirmative"})
_NO_TOKENS = frozenset({"no", "nope", "nah", "false", "incorrect", "negative"})

#: Words that flip a claim. Used to stop "the car is red" from matching
#: "the car is not red".
_NEGATIONS = frozenset({"not", "isnt", "arent", "doesnt", "dont", "cannot", "cant", "never", "without", "neither", "nor"})

#: Mutually exclusive answers. If gold says one and pred says the other, it is wrong
#: no matter how much text they share. This is the guard that makes containment safe
#: on the relational questions ("is X on the left or right of Y?").
_ANTONYMS = [
    {"left", "right"},
    {"above", "below"},
    {"top", "bottom"},
    {"up", "down"},
    {"front", "behind"},
    {"inside", "outside"},
    {"open", "closed"},
    {"on", "off"},
    {"near", "far"},
    {"first", "last"},
    {"more", "less"},
    {"bigger", "smaller"},
    {"larger", "smaller"},
    {"taller", "shorter"},
]

#: Tokens too generic to carry a match on their own.
_STOPWORDS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "as", "that", "this", "these", "those",
        "it", "its", "and", "or", "but", "there", "here", "side", "picture",
        "image", "photo", "appears", "appear", "looks", "look", "seems", "seem",
        "has", "have", "had", "does", "do", "did", "indeed", "color", "colour",
    }
)


def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    The standard VQA/SQuAD normalization. Everything else in this module compares
    normalized strings, never raw ones.
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_PUNCT)
    raw = [t for t in s.split() if t]
    toks = [t for t in raw if t not in _ARTICLES]
    # Never let article-stripping empty the string. A bare "A" is the multiple-choice
    # option letter, not an article, and dropping it silently scored every 'A' answer
    # on V*Bench as wrong.
    return " ".join(toks or raw)


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def _polarity(s: str) -> str | None:
    """'yes', 'no', or None. Reads the FIRST token only, or the whole string.

    Position matters. "No, the car is not red" -> 'no'. "not sure" -> None, because
    "not" is not a polarity token and "no" is never matched as a prefix of a word.
    """
    toks = _tokens(s)
    if not toks:
        return None
    if toks[0] in _YES_TOKENS:
        return "yes"
    if toks[0] in _NO_TOKENS:
        return "no"
    return None


def _has_negation(s: str) -> bool:
    return any(t in _NEGATIONS for t in _tokens(s))


def _antonym_conflict(a: str, b: str) -> bool:
    """True if a and b commit to opposite sides of the same axis."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    for pair in _ANTONYMS:
        x, y = tuple(pair)
        if (x in ta and y in tb and x not in tb and y not in ta) or (
            y in ta and x in tb and y not in tb and x not in ta
        ):
            return True
    return False


def _whole_word_in(needle: str, haystack: str) -> bool:
    """Whole-word containment on normalized text.

    Word boundaries are the point. "no" must not match inside "not", and "red" must
    not match inside "covered".
    """
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def _content_tokens(s: str) -> list[str]:
    return [t for t in _tokens(s) if t not in _STOPWORDS]


# --- answer shape -------------------------------------------------------------

#: "The color of the shirt is white." / "The dog is to the left of the bottle."
_TEMPLATE = re.compile(r"^(?P<subj>.+?)\s+(?:is|are|was|were|has|have|appears?|wears?)\s+(?P<tail>.+)$")

#: How long the extracted tail may be and still count as a closed-form answer.
SHORT_TAIL_WORDS = 3

#: How long a whole gold may be and still count as a short answer.
SHORT_ANSWER_WORDS = 4

#: A widened gold must still be a single graspable claim, not a paragraph.
WIDE_MAX_WORDS = 20


def answer_tail(gold: str) -> str:
    """The discriminating span of a declarative gold.

    "The color of the shirt is white." -> "white"
    "The dog is to the left of the bottle." -> "to left of bottle"

    Returns the normalized gold unchanged when it matches no template.
    """
    norm = normalize_answer(gold)
    m = _TEMPLATE.match(norm)
    if not m:
        return norm
    tail = m.group("tail").strip()
    # Strip a leading negation so "is not red" yields "red"; polarity is handled
    # separately and must not leak into the content span.
    toks = tail.split()
    while toks and toks[0] in _NEGATIONS:
        toks = toks[1:]
    return " ".join(toks) if toks else norm


def answer_shape(gold: str) -> str:
    """Coarse bucket, for the Gate 0.5 report. One of:

    'yes_no', 'single_word', 'number', 'short_phrase', 'attribute_sentence',
    'relational_sentence', 'long_sentence'.
    """
    norm = normalize_answer(gold)
    toks = norm.split()
    if not toks:
        return "long_sentence"
    if _polarity(gold) is not None:
        return "yes_no"
    if len(toks) == 1:
        return "number" if _is_number(toks[0]) else "single_word"
    if all(_is_number(t) for t in toks):
        return "number"
    if len(toks) <= SHORT_ANSWER_WORDS:
        return "short_phrase"
    tail = answer_tail(gold)
    ttoks = tail.split()
    if _TEMPLATE.match(norm) and len(ttoks) <= SHORT_TAIL_WORDS:
        return "attribute_sentence"
    if _TEMPLATE.match(norm) and len(toks) <= WIDE_MAX_WORDS:
        return "relational_sentence"
    return "long_sentence"


def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", tok))


def answer_is_scorable(gold: str, widened: bool = False) -> bool:
    """The Gate 0.5 filter. True if a deterministic checker can grade this gold.

    STRICT (default) keeps golds that reduce to a closed form:
      * yes/no polarity ("Yes, the color of the pole is gray." -> yes),
      * a bare short answer (<= 4 words),
      * a declarative whose tail is <= 3 words ("...is white." -> "white").

    WIDENED additionally keeps single-claim declaratives whose tail is longer --
    the relational answers ("The dog is to the left of the bottle."). Those are
    graded by normalized whole-word containment with an antonym guard, which GOAL
    S9 prescribes as the fallback. Multi-clause and runaway golds stay rejected;
    nothing here can grade them without a judge.
    """
    if not gold or not gold.strip():
        return False
    shape = answer_shape(gold)
    if shape in ("yes_no", "single_word", "number", "short_phrase", "attribute_sentence"):
        return True
    if widened and shape == "relational_sentence":
        # Needs at least one content token to match on, and one clause only.
        return len(_content_tokens(answer_tail(gold))) >= 1 and normalize_answer(gold).count(" and ") == 0
    return False


# --- scoring ------------------------------------------------------------------

#: A prediction that is nothing but an option letter: "B", "(B)", "B.", " b ".
_BARE_LETTER = re.compile(r"^\s*[(\[]?\s*([A-Za-z])\s*[)\].:,]?\s*$")
#: A letter announced in a sentence: "Answer: B", "The answer is (C)", "Option D".
_NAMED_LETTER = re.compile(r"\b(?:answer|option|choice)\b\W{0,4}(?:is\b\W{0,4})?[(\[]?([A-Za-z])[)\].:,]?(?:\s|$)", re.I)


def _option_letter(pred: str, n_options: int) -> int | None:
    """Index of the option a letter-style prediction names, or None.

    Read off the RAW string, before normalization. "A" is both a valid option letter
    and an English article, and only the raw form still tells them apart.
    """
    if not pred or n_options <= 0:
        return None
    for rx in (_BARE_LETTER, _NAMED_LETTER):
        m = rx.search(pred.strip())
        if m:
            idx = ord(m.group(1).lower()) - ord("a")
            if 0 <= idx < n_options:
                return idx
    return None


def _best_option(pred: str, options: list[str]) -> str | None:
    """The option the prediction is pointing at, or None if nothing fits.

    Accepts a letter ("B", "(B)", "Answer: B"), the option text, or a fragment of it.
    """
    idx = _option_letter(pred, len(options))
    if idx is not None:
        return options[idx]

    npred = normalize_answer(pred)
    if not npred:
        return None

    ptoks = npred.split()
    best, best_score = None, 0.0
    for opt in options:
        nopt = normalize_answer(opt)
        if not nopt:
            continue
        if npred == nopt:
            return opt
        otoks = set(_content_tokens(opt))
        if not otoks:
            continue
        overlap = otoks & set(_content_tokens(pred))
        score = len(overlap) / len(otoks)
        # Whole-option containment is a strong signal either way.
        if _whole_word_in(nopt, npred) or (len(ptoks) <= SHORT_ANSWER_WORDS and _whole_word_in(npred, nopt)):
            score += 1.0
        if score > best_score:
            best, best_score = opt, score
    # Below this the "match" is one stray shared word; call it no answer.
    return best if best_score >= 0.5 else None


def answer_correct(pred: str, gold: str, options: list[str] | None = None) -> bool:
    """Deterministic replacement for the DeepEyes LLM judge (GOAL S9, G0.5).

    Order of resolution:
      1. normalized exact match,
      2. multiple choice -- snap `pred` to an option, compare that to `gold`,
      3. yes/no golds -- compare polarity, with a content-clause fallback,
      4. containment -- one side is a whole-word span of the other and the short
         side is <= 4 words, guarded by the antonym and negation checks.
    """
    if pred is None or gold is None:
        return False
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    if not ngold:
        return False

    # 1. multiple choice: the option is the unit of comparison. This runs first so a
    # bare letter answer is resolved before any text heuristic sees it.
    if options:
        chosen = _best_option(pred, options)
        if chosen is None:
            return False
        return normalize_answer(chosen) == ngold

    if not npred:
        return False

    # 2. exact
    if npred == ngold:
        return True

    # Opposite sides of the same axis is a hard no, whatever else matches.
    if _antonym_conflict(npred, ngold):
        return False

    # 3. yes/no golds
    gold_pol = _polarity(gold)
    if gold_pol is not None:
        pred_pol = _polarity(pred)
        if pred_pol is not None:
            if pred_pol != gold_pol:
                return False
            # Polarity agrees. If the prediction adds a claim, it must not
            # contradict the gold's claim.
            gtail = answer_tail(" ".join(_tokens(gold)[1:])) if len(_tokens(gold)) > 1 else ""
            if gtail and len(_content_tokens(gtail)) >= 1:
                if _antonym_conflict(npred, gtail):
                    return False
            return True
        # No explicit yes/no in the prediction. Only a 'Yes' gold can be recovered,
        # by matching the asserted content; a 'No' gold needs the denial stated.
        if gold_pol == "no":
            return False
        gtail = answer_tail(" ".join(_tokens(gold)[1:]))
        if not gtail or not _content_tokens(gtail):
            return False
        if _has_negation(npred):
            return False
        return _whole_word_in(gtail, npred) or (
            len(npred.split()) <= SHORT_ANSWER_WORDS and _whole_word_in(npred, gtail)
        )

    # 4. declarative gold. Compare on the discriminating tail.
    gtail = answer_tail(gold)
    ptail = answer_tail(pred)
    if gtail and ptail and gtail == ptail:
        return not (_has_negation(npred) ^ _has_negation(ngold))
    # A negated prediction cannot satisfy an affirmative gold.
    if _has_negation(npred) != _has_negation(ngold):
        return False

    for needle, hay in ((ngold, npred), (npred, ngold)):
        other = npred if needle is ngold else ngold
        if len(needle.split()) > SHORT_ANSWER_WORDS and needle != gtail:
            continue
        if _whole_word_in(needle, hay):
            return True
        del other

    # Short prediction against the gold's tail, e.g. pred "white" vs
    # gold "The color of the shirt is white."
    if gtail:
        if len(npred.split()) <= SHORT_ANSWER_WORDS and _content_tokens(npred):
            if _whole_word_in(npred, gtail):
                return True
        if len(gtail.split()) <= SHORT_ANSWER_WORDS and _content_tokens(gtail):
            if _whole_word_in(gtail, npred):
                return True
    return False


# --- TRAIN --------------------------------------------------------------------

_LIGHT_COLUMNS = ["reward_model", "extra_info", "data_source"]


def _train_index(seed: int, filtered_only: bool, widened: bool = False) -> tuple[list[int], list[dict]]:
    """Read the light columns only, filter, shuffle. Never touches the image bytes.

    The image column is 990 MB of inline JPEG in a single row group. Deciding WHICH
    rows we want before reading any of it is what keeps this fast.
    """
    import pyarrow.parquet as pq

    _require(TRAIN_PARQUET, "DeepEyes train parquet")
    pf = pq.ParquetFile(TRAIN_PARQUET)
    tbl = pf.read(columns=_LIGHT_COLUMNS)
    rm = tbl.column("reward_model").to_pylist()
    ei = tbl.column("extra_info").to_pylist()

    rows = []
    for i, (r, e) in enumerate(zip(rm, ei)):
        gold = (r or {}).get("ground_truth") or ""
        question = (e or {}).get("question") or ""
        if not gold or not question:
            continue
        if filtered_only and not answer_is_scorable(gold, widened=widened):
            continue
        rows.append({"row": i, "gold": gold, "question": question.strip(), "index": (e or {}).get("index")})

    random.Random(seed).shuffle(rows)
    return [r["row"] for r in rows], rows


def load_train(limit: int | None = None, seed: int = 0, filtered_only: bool = True) -> list[Sample]:
    """DeepEyes train split as `Sample`s, deterministically shuffled.

    `Sample.image` is the ORIGINAL full-resolution RGB image -- the zoom env owns
    every resize (contract.py, GOAL S18). `Sample.question` is the raw question with
    no `<image>` marker and no format scaffold; the harness builds the prompt.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    _, rows = _train_index(seed=seed, filtered_only=filtered_only)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return []

    wanted = {r["row"]: pos for pos, r in enumerate(rows)}
    stop_after = max(wanted)

    out: list[Sample | None] = [None] * len(rows)
    pf = pq.ParquetFile(TRAIN_PARQUET)
    offset = 0
    # Stream the image column so peak memory stays at one batch, not 990 MB.
    for batch in pf.iter_batches(batch_size=256, columns=["images"]):
        n = batch.num_rows
        if offset + n <= stop_after + 1 or offset <= stop_after:
            imgs = batch.column("images")
            for j in range(n):
                gi = offset + j
                if gi not in wanted:
                    continue
                cell = imgs[j].as_py()
                if not cell:
                    continue
                img = Image.open(io.BytesIO(cell[0]["bytes"])).convert("RGB")
                r = rows[wanted[gi]]
                out[wanted[gi]] = Sample(
                    sid=f"deepeyes-{r['row']}",
                    image=img,
                    question=r["question"],
                    gold=r["gold"],
                    source="deepeyes",
                    options=None,
                    meta={
                        "row": r["row"],
                        "shape": answer_shape(r["gold"]),
                        "polarity": _polarity(r["gold"]),
                        "orig_index": r["index"],
                        "path": (cell[0].get("path") if isinstance(cell[0], dict) else None),
                    },
                )
        offset += n
        if offset > stop_after:
            break

    return [s for s in out if s is not None]


def balance_polarity(samples: list[Sample], seed: int = 0) -> list[Sample]:
    """Downsample the majority yes/no class to 50/50. OPTIONAL, but read this.

    Measured over all 22,362 train rows: **82.4% of the yes/no golds are "Yes"**
    (9,125 yes vs 1,944 no), and yes/no golds are 49.5% of the corpus. So "always
    say yes" scores ~0.82 on half the data without looking at the image.

    Two things break if you ignore it. Gate 1 reads a difficulty band off
    thumbnail-only accuracy -- the band comes out high for a reason that has nothing
    to do with resolution. And GRPO gets a reward the policy can farm without ever
    calling the zoom tool, which is exactly the behaviour this project exists to
    measure. Non-yes/no samples pass through untouched.
    """
    yes = [s for s in samples if s.meta.get("polarity") == "yes"]
    no = [s for s in samples if s.meta.get("polarity") == "no"]
    rest = [s for s in samples if s.meta.get("polarity") is None]
    k = min(len(yes), len(no))
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    out = yes[:k] + no[:k] + rest
    rng.shuffle(out)
    return out


# --- EVAL (V*Bench) -----------------------------------------------------------

def load_vstar(limit: int | None = None) -> list[Sample]:
    """V*Bench eval split, 191 items, deterministic order.

    **The options are shuffled.** In the raw benchmark `options[0]` is the gold in
    191/191 files -- verified against the `label` column of `test_questions.jsonl`
    (see `.notes/vstar_gold.md`). Serving them in file order would let a model score
    100% by always picking the first one, so each sample gets a fixed permutation
    derived from its sid: same sid, same order, every run, no seed argument needed.

    `Sample.gold` is the gold option STRING. `Sample.meta["gold_index"]` is its
    position in the shuffled `Sample.options`, and `meta["gold_letter"]` the matching
    A/B/C/D.
    """
    from PIL import Image

    _require(VSTAR_ROOT, "V*Bench eval set")

    paths: list[Path] = []
    for sub in VSTAR_SUBDIRS:
        d = VSTAR_ROOT / sub
        if not d.is_dir():
            continue
        paths.extend(sorted(d.glob("sa_*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No V*Bench json under {VSTAR_ROOT}/{{{','.join(VSTAR_SUBDIRS)}}}."
        )

    out: list[Sample] = []
    for jp in paths:
        ip = jp.with_suffix(".jpg")
        if not ip.exists():
            continue
        with open(jp) as fh:
            meta = json.load(fh)
        options = list(meta.get("options") or [])
        question = (meta.get("question") or "").strip()
        if not options or not question:
            continue

        gold = options[0]  # verified: raw file order puts the gold first
        sid = f"vstar-{jp.parent.name}-{jp.stem}"
        shuffled = list(options)
        random.Random(sid).shuffle(shuffled)
        gold_index = shuffled.index(gold)

        out.append(
            Sample(
                sid=sid,
                image=Image.open(ip).convert("RGB"),
                question=question,
                gold=gold,
                source="vstar",
                options=shuffled,
                meta={
                    "gold_index": gold_index,
                    "gold_letter": chr(ord("A") + gold_index),
                    "category": jp.parent.name,
                    "target_object": meta.get("target_object"),
                    "bbox": meta.get("bbox"),
                    "image_path": str(ip),
                    "n_options": len(shuffled),
                },
            )
        )

    out.sort(key=lambda s: s.sid)
    return out[:limit] if limit is not None else out


# --- self-checks --------------------------------------------------------------

def _selfcheck() -> int:
    cases: list[tuple[str, str, list[str] | None, bool]] = [
        # --- exact and trivial ---
        ("white", "white", None, True),
        ("White.", "white", None, True),
        ("The color of the shirt is white.", "The color of the shirt is white.", None, True),
        # --- yes/no polarity ---
        ("Yes", "Yes, the color of the pole is gray.", None, True),
        ("yes, it is gray", "Yes, the color of the pole is gray.", None, True),
        ("No", "Yes, the color of the pole is gray.", None, False),
        ("No, the car is not red.", "No, the car is not on the left side of the person.", None, True),
        ("Yes, the car is red.", "No, the car is not red.", None, False),
        # negation trap: "no" must not be read out of "not sure"
        ("not sure", "No, the car is not red.", None, False),
        ("nothing", "No, the car is not red.", None, False),
        # a 'No' gold needs the denial stated, not just the words echoed
        ("the car is red", "No, the car is not red.", None, False),
        # --- declarative tails ---
        ("white", "The color of the shirt is white.", None, True),
        ("The shirt is white.", "The color of the shirt is white.", None, True),
        ("black", "The color of the shirt is white.", None, False),
        ("plaid", "The pattern of the shirt is plaid.", None, True),
        # --- antonym traps ---
        ("The dog is to the right of the bottle.", "The dog is to the left of the bottle.", None, False),
        ("The dog is to the left of the bottle.", "The dog is to the left of the bottle.", None, True),
        ("left", "The clock is on the left side of the cake.", None, True),
        ("right", "The clock is on the left side of the cake.", None, False),
        ("Yes, the umbrella is on the left side of the traffic light.",
         "Yes, the umbrella is on the right side of the traffic light.", None, False),
        # --- word-boundary traps ---
        ("The wall is covered.", "The color of the wall is red.", None, False),
        ("nowhere", "No.", None, False),
        # --- multiple choice ---
        ("B", "The color of the dustpan is red.",
         ["The color of the dustpan is blue.", "The color of the dustpan is red."], True),
        ("A", "The color of the dustpan is red.",
         ["The color of the dustpan is blue.", "The color of the dustpan is red."], False),
        ("The color of the dustpan is red.", "The color of the dustpan is red.",
         ["The color of the dustpan is blue.", "The color of the dustpan is red."], True),
        ("red", "The color of the dustpan is red.",
         ["The color of the dustpan is blue.", "The color of the dustpan is red."], True),
        ("purple", "The color of the dustpan is red.",
         ["The color of the dustpan is blue.", "The color of the dustpan is red."], False),
        # option letter 'A' must not be eaten as an English article
        ("A", "The color of the flag is white.",
         ["The color of the flag is white.", "The color of the flag is red."], True),
        ("(A)", "The color of the flag is white.",
         ["The color of the flag is white.", "The color of the flag is red."], True),
        ("Answer: B", "The color of the flag is red.",
         ["The color of the flag is white.", "The color of the flag is red."], True),
        ("The answer is (B).", "The color of the flag is white.",
         ["The color of the flag is white.", "The color of the flag is red."], False),
        ("a", "The color of the flag is white.",
         ["The color of the flag is white.", "The color of the flag is red."], True),
        # --- junk ---
        ("", "white", None, False),
        ("I cannot tell.", "The color of the shirt is white.", None, False),
    ]

    fails = 0
    for pred, gold, opts, want in cases:
        got = answer_correct(pred, gold, opts)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            fails += 1
            print(f"  {flag} pred={pred!r} gold={gold!r} opts={bool(opts)} want={want} got={got}")
    print(f"answer_correct: {len(cases) - fails}/{len(cases)} passed")

    shape_cases = [
        ("Yes, the color of the pole is gray.", "yes_no", True),
        ("No, the car is not on the left side of the person.", "yes_no", True),
        ("The color of the shirt is white.", "attribute_sentence", True),
        ("The dog is to the left of the bottle.", "relational_sentence", False),
        ("white", "single_word", True),
        ("3", "number", True),
        ("a red brick wall", "short_phrase", True),
    ]
    sfails = 0
    for gold, want_shape, want_scorable in shape_cases:
        gs, sc = answer_shape(gold), answer_is_scorable(gold)
        if gs != want_shape or sc != want_scorable:
            sfails += 1
            print(f"  FAIL {gold!r} shape={gs} (want {want_shape}) scorable={sc} (want {want_scorable})")
    print(f"answer_shape/answer_is_scorable: {len(shape_cases) - sfails}/{len(shape_cases)} passed")

    # normalize_answer
    nfails = 0
    for raw, want in [
        ("The color of the shirt is white.", "color of shirt is white"),
        ("  YES,  it is.  ", "yes it is"),
        ("(B) red", "b red"),
    ]:
        if normalize_answer(raw) != want:
            nfails += 1
            print(f"  FAIL normalize_answer({raw!r}) = {normalize_answer(raw)!r} want {want!r}")
    print(f"normalize_answer: {3 - nfails}/3 passed")

    total = fails + sfails + nfails
    print("ALL PASS" if total == 0 else f"{total} FAILURES")
    return total


if __name__ == "__main__":
    import sys

    sys.exit(1 if _selfcheck() else 0)
