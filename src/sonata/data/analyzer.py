"""
analyzer.py
===========
Genre-agnostic, reusable MIDI harmonic analysis engine.

Main class
----------
MIDIHarmonicAnalyzer
    analyze(source)         → report dict (chord progression + key info)
    extract_features(report) → OrderedDict (flat feature vector)

Usage example
-------------
>>> from sonata.data.analyzer import MIDIHarmonicAnalyzer
>>> analyzer = MIDIHarmonicAnalyzer()
>>> report   = analyzer.analyze("path/to/file.mid")
>>> features = analyzer.extract_features(report)
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import music21 as m21

__all__ = ["MIDIHarmonicAnalyzer"]


class MIDIHarmonicAnalyzer:
    """
    Analyse any MIDI file (or music21 Score) and extract:

    * Beat-level chord progression — Harte label, Roman numeral, functional tag
    * Key / modality detection (embedded key markers + algorithmic fallback)
    * Chord-transition probability matrix (1st-order Markov)
    * Genre-level feature vector suitable for machine-learning pipelines

    Parameters
    ----------
    key_window : int
        Number of measures used as context window for algorithmic key detection.
    key_confidence : float
        Minimum correlation coefficient required before flagging an
        algorithmic key result as uncertain (appends '*' to key_source tag).
    """

    # ── quality → Harte shorthand ────────────────────────────────────
    QUALITY_TO_HARTE: Dict[str, str] = {
        "major":                    "maj",
        "minor":                    "min",
        "diminished":               "dim",
        "augmented":                "aug",
        "dominant seventh":         "7",
        "major seventh":            "maj7",
        "minor seventh":            "min7",
        "half-diminished seventh":  "hdim7",
        "diminished seventh":       "dim7",
        "augmented sixth":          "aug6",
        "augmented major seventh":  "augmaj7",
        "other":                    "other",
    }

    # ── scale-degree → functional tag ────────────────────────────────
    FUNCTION_MAP:   Dict[int, str] = {1: "T", 2: "PD", 3: "T", 4: "S", 5: "D", 6: "PD", 7: "D"}
    FUNCTION_NAMES: Dict[str, str] = {"T": "Tonic", "D": "Dominant", "S": "Subdominant", "PD": "Predominant"}

    # ── regex helpers for cleaning Roman-numeral figures ─────────────
    _INV_JUNK  = re.compile(r"(?<=[IViv])[#b]?\d{2,}")
    _DYAD_JUNK = re.compile(r"[#b]?\d+$")

    def __init__(self, key_window: int = 6, key_confidence: float = 0.80) -> None:
        self.key_window     = key_window
        self.key_confidence = key_confidence

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def analyze(self, source, verbose: bool = False) -> Dict:
        """
        Analyse a MIDI file path or a music21 Score.

        Parameters
        ----------
        source : str | music21.stream.Score
            Path to a MIDI file, or a pre-parsed music21 Score.
        verbose : bool
            If True, print a formatted chord-by-chord report.

        Returns
        -------
        dict with keys:
            chords        — list of per-beat chord record dicts
            global_key    — music21 Key object for the whole piece
            keys_visited  — ordered list of tonal centre name strings
            modulations   — list of (measure, beat, old_key, new_key) tuples
            num_measures  — total measure count (int)
        """
        if isinstance(source, str):
            score = m21.converter.parse(source)
        else:
            score = source

        chordified = score.chordify()
        measured   = chordified.makeMeasures()
        measures   = list(measured.getElementsByClass("Measure"))

        emb_keys, emb_offsets = self._collect_embedded_keys(score)
        global_key = score.analyze("key")

        prev_key_name = None
        chords: List[Dict] = []

        for i, meas in enumerate(measures):
            m_num = meas.number if meas.number else (i + 1)
            for ch in meas.recurse().getElementsByClass("Chord"):
                abs_off = meas.offset + ch.offset

                local_key, key_src = self._resolve_key(
                    i, abs_off, measures, emb_keys, emb_offsets
                )
                key_name = local_key.name

                mod = None
                if prev_key_name and key_name != prev_key_name:
                    mod = (prev_key_name, key_name)
                prev_key_name = key_name

                closed  = ch.closedPosition()
                is_dyad = len(ch.pitches) == 2
                harte   = self._harte_label(closed)

                try:
                    rn    = m21.roman.romanNumeralFromChord(closed, local_key)
                    roman = self._clean_roman(rn.figure, is_dyad)
                    func  = self.FUNCTION_MAP.get(rn.scaleDegree, "?")
                except Exception:
                    roman, func = "?", "?"

                rec = OrderedDict(
                    measure      = m_num,
                    beat         = round(ch.offset, 2),
                    abs_offset   = round(abs_off, 2),
                    duration     = round(ch.quarterLength, 2),
                    key          = key_name,
                    key_source   = key_src,
                    key_mode     = local_key.mode,
                    harte        = harte,
                    roman        = roman,
                    function     = func,
                    function_name= self.FUNCTION_NAMES.get(func, "Unknown"),
                    is_dyad      = is_dyad,
                    num_pitches  = len(ch.pitches),
                    pitch_classes= sorted({p.pitchClass for p in ch.pitches}),
                    modulation   = mod,
                )
                chords.append(rec)

        keys_visited = list(dict.fromkeys(r["key"] for r in chords))
        modulations  = [
            (chords[j]["measure"], chords[j]["beat"],
             chords[j - 1]["key"], chords[j]["key"])
            for j in range(1, len(chords))
            if chords[j]["key"] != chords[j - 1]["key"]
        ]

        report = {
            "chords":       chords,
            "global_key":   global_key,
            "keys_visited": keys_visited,
            "modulations":  modulations,
            "num_measures": len(measures),
        }

        if verbose:
            self._print_report(report)

        return report

    def extract_features(self, report: Dict) -> Dict:
        """
        Compute a flat feature vector from an analysis report.

        Returns an OrderedDict suitable as a single row for pd.DataFrame.
        Internal objects (transition matrix, top lists) are prefixed with '_'
        or stored as serialisable types.
        """
        chords = report["chords"]
        if not chords:
            return {}

        romans    = [c["roman"]    for c in chords if c["roman"]    != "?"]
        funcs     = [c["function"] for c in chords]
        durations = [c["duration"] for c in chords if c["duration"] > 0]
        n_pitches = [c["num_pitches"] for c in chords]

        feat: Dict = OrderedDict()

        # ── 1. Key & modality ──────────────────────────────────────────
        feat["global_key"]              = report["global_key"].name
        feat["global_mode"]             = report["global_key"].mode
        feat["num_keys"]                = len(report["keys_visited"])
        feat["num_modulations"]         = len(report["modulations"])
        feat["modulations_per_measure"] = (
            len(report["modulations"]) / max(report["num_measures"], 1)
        )
        major_beats = sum(1 for c in chords if c["key_mode"] == "major")
        feat["major_mode_ratio"] = major_beats / max(len(chords), 1)

        # ── 2. Chord vocabulary ────────────────────────────────────────
        unique_romans = set(romans)
        feat["chord_vocab_roman"]  = len(unique_romans)
        feat["chord_vocab_harte"]  = len({c["harte"] for c in chords})
        feat["unique_chord_ratio"] = len(unique_romans) / max(len(romans), 1)

        # ── 3. Functional distribution ────────────────────────────────
        func_counts = Counter(funcs)
        total_f = max(sum(func_counts.values()), 1)
        for tag in ("T", "D", "S", "PD"):
            feat[f"func_ratio_{tag}"] = func_counts.get(tag, 0) / total_f
        feat["func_unknown_ratio"] = func_counts.get("?", 0) / total_f

        # ── 4. Harmonic rhythm ─────────────────────────────────────────
        if durations:
            feat["harm_rhythm_mean"] = float(np.mean(durations))
            feat["harm_rhythm_std"]  = float(np.std(durations))
            feat["harm_rhythm_min"]  = float(np.min(durations))
            feat["harm_rhythm_max"]  = float(np.max(durations))
        else:
            feat["harm_rhythm_mean"] = feat["harm_rhythm_std"] = 0.0
            feat["harm_rhythm_min"]  = feat["harm_rhythm_max"] = 0.0

        # ── 5. Chord complexity ────────────────────────────────────────
        feat["avg_chord_cardinality"] = float(np.mean(n_pitches))
        feat["max_chord_cardinality"] = int(np.max(n_pitches))
        feat["dyad_ratio"] = sum(1 for c in chords if c["is_dyad"]) / max(len(chords), 1)

        # ── 6. Transition matrix (1st-order Markov) ───────────────────
        tm, labels = self._build_transition_matrix(romans)
        feat["_transition_matrix"] = tm
        feat["_transition_labels"] = labels

        if tm is not None and tm.size > 0:
            flat = tm[tm > 0]
            feat["transition_entropy"] = float(-np.sum(flat * np.log2(flat)))
        else:
            feat["transition_entropy"] = 0.0

        # ── 7. Interval-class vector (Forte-style) ────────────────────
        feat["interval_class_vector"] = self._interval_class_vector(chords)

        # ── 8. Top bigrams & chords ───────────────────────────────────
        bigram_counts: Counter = Counter()
        for a, b in zip(romans[:-1], romans[1:]):
            bigram_counts[(a, b)] += 1
        feat["top_bigrams"] = bigram_counts.most_common(10)
        feat["top_chords"]  = Counter(romans).most_common(10)

        return feat

    # ─────────────────────────────────────────────────────────────────────
    #  STATIC HELPERS
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_transition_matrix(
        roman_sequence: List[str],
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Row-normalised first-order Markov transition matrix over Roman labels."""
        if len(roman_sequence) < 2:
            return None, []
        labels = sorted(set(roman_sequence))
        idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        counts = np.zeros((n, n), dtype=float)
        for a, b in zip(roman_sequence[:-1], roman_sequence[1:]):
            counts[idx[a], idx[b]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return counts / row_sums, labels

    @staticmethod
    def _interval_class_vector(chords: List[Dict]) -> List[int]:
        """Aggregate Forte interval-class vector (ic1–ic6) across all chords."""
        icv = [0] * 6
        for c in chords:
            pcs = c["pitch_classes"]
            for i in range(len(pcs)):
                for j in range(i + 1, len(pcs)):
                    diff = abs(pcs[i] - pcs[j])
                    ic = min(diff, 12 - diff)
                    if 1 <= ic <= 6:
                        icv[ic - 1] += 1
        return icv

    @staticmethod
    def _collect_embedded_keys(score) -> Tuple[Dict, List]:
        """Return {offset: Key} from explicit Key / KeySignature objects in the score."""
        emb: Dict = {}
        for part in score.parts:
            for el in part.recurse():
                if isinstance(el, m21.key.Key):
                    off = el.getOffsetInHierarchy(score)
                    emb[off] = el
                elif isinstance(el, m21.key.KeySignature):
                    off = el.getOffsetInHierarchy(score)
                    emb[off] = el.asKey()
        return emb, sorted(emb.keys())

    def _resolve_key(
        self,
        measure_idx: int,
        abs_offset: float,
        measures,
        emb_keys: Dict,
        emb_offsets: List,
    ):
        """Return (Key, source_tag) — prefer embedded markers, fall back to analysis."""
        best = None
        for o in emb_offsets:
            if o <= abs_offset:
                best = emb_keys[o]
            else:
                break
        if best is not None:
            return best, "embedded"

        half  = self.key_window // 2
        start = max(0, measure_idx - half)
        end   = min(len(measures), measure_idx + half + 1)
        window = m21.stream.Stream(measures[start:end])
        algo   = window.analyze("key")
        conf   = algo.correlationCoefficient
        tag    = f"algo({conf:.2f})" + ("" if conf >= self.key_confidence else "*")
        return algo, tag

    def _harte_label(self, chord_obj) -> str:
        root  = chord_obj.root().name
        q     = chord_obj.quality
        if chord_obj.isSeventh():
            q = getattr(chord_obj, "seventhString", q)
        short = self.QUALITY_TO_HARTE.get(q, q)
        if len(chord_obj.pitches) == 2 and short == "other":
            iv    = m21.interval.Interval(chord_obj.pitches[0], chord_obj.pitches[1])
            short = f"interval({iv.simpleName})"
        return f"{root}:{short}"

    def _clean_roman(self, figure: str, is_dyad: bool = False) -> str:
        c = self._INV_JUNK.sub("", figure)
        if is_dyad:
            c = self._DYAD_JUNK.sub("", c)
        return c.strip() or figure

    # ─────────────────────────────────────────────────────────────────────
    #  DISPLAY HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _print_report(self, report: Dict) -> None:
        chords = report["chords"]
        print(f"\n{'═' * 95}")
        print(
            f"  Global key : {report['global_key'].name}  |  "
            f"Measures: {report['num_measures']}  |  "
            f"Chords: {len(chords)}  |  "
            f"Modulations: {len(report['modulations'])}"
        )
        print(f"  Keys visited: {' → '.join(report['keys_visited'])}")
        print(f"{'═' * 95}\n")

        prev_m = None
        for r in chords:
            mod_s = f"  ◀ MOD: {r['modulation'][0]}→{r['modulation'][1]}" if r["modulation"] else ""
            dy    = " ◇" if r["is_dyad"] else ""
            if r["measure"] != prev_m:
                if prev_m is not None:
                    print(f"  {'─' * 90}")
                prev_m = r["measure"]
            print(
                f"  M{r['measure']:<3} b{r['beat']:<5} │ "
                f"{r['key']:<10} │ {r['harte']:<18} │ "
                f"{r['roman']:<8} │ {r['function']:<2} "
                f"({r['function_name']:<13}){dy}{mod_s}"
            )
        print(f"  {'─' * 90}\n")

    def print_features(self, features: Dict) -> None:
        """Pretty-print the flat feature vector."""
        skip = {"_transition_matrix", "_transition_labels", "top_bigrams", "top_chords", "interval_class_vector"}
        print(f"\n{'═' * 70}")
        print("  GENRE FEATURE VECTOR")
        print(f"{'═' * 70}")
        for k, v in features.items():
            if k in skip:
                continue
            print(f"  {k:<30s}: {v:.4f}" if isinstance(v, float) else f"  {k:<30s}: {v}")
        print(f"\n  Interval-class vector (ic1–ic6): {features.get('interval_class_vector', [])}")
        print("\n  Top 10 chords (Roman):")
        for chord, cnt in features.get("top_chords", []):
            print(f"    {chord:<10s}  ×{cnt}")
        print("\n  Top 10 chord transitions:")
        for (a, b), cnt in features.get("top_bigrams", []):
            print(f"    {a:<8} → {b:<8}  ×{cnt}")
        print()
