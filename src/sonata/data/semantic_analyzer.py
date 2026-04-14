"""
semantic_analyzer.py
====================
Semantic harmonic feature extraction using **musif** and **music21**.

This module forms the **semantic branch** of the dual-branch feature
extraction pipeline.  It extracts musically meaningful features:

* Roman-numeral chord labels  (relative to the detected key)
* Functional-harmonic ratios  (T / D / S / PD)
* Key & mode analysis
* Chord-progression bigrams
* Cadence detection
* Tension / resolution profiles

Main class
----------
SemanticAnalyzer
    analyze(midi_path)       → SemanticReport   (rich dict)
    extract_features(report) → dict[str, Any]   (flat feature vector)
"""

from __future__ import annotations

import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sonata.config.settings import CFG

__all__ = ["SemanticAnalyzer", "SemanticReport"]

# ── Functional-harmony classification (simplified) ──────────────────────────
_ROMAN_FUNCTIONS = {
    # Tonic function
    "I": "T", "i": "T", "VI": "T", "vi": "T", "III": "T", "iii": "T",
    # Dominant function
    "V": "D", "v": "D", "VII": "D", "vii": "D", "viio": "D",
    # Subdominant function
    "IV": "S", "iv": "S", "II": "S", "ii": "S",
    # Pre-dominant
    "iio": "PD", "IV6": "PD", "ii6": "PD",
}

# Common cadence patterns (final two chords)
_CADENCE_PATTERNS = {
    ("V", "I"):    "authentic",
    ("V", "i"):    "authentic",
    ("V7", "I"):   "authentic",
    ("vii", "I"):  "authentic",
    ("IV", "I"):   "plagal",
    ("iv", "i"):   "plagal",
    ("V", "vi"):   "deceptive",
    ("V", "VI"):   "deceptive",
    ("I", "V"):    "half",
    ("ii", "V"):   "half",
    ("IV", "V"):   "half",
}


@dataclass
class SemanticReport:
    """Container for the output of SemanticAnalyzer.analyze()."""
    file_path: str
    global_key: str = ""
    global_mode: str = ""
    key_confidence: float = 0.0

    # Chord sequence (Roman-numeral labels)
    roman_numerals: List[str] = field(default_factory=list)

    # Raw chord objects (Harte notation or music21 labels)
    chord_labels: List[str] = field(default_factory=list)
    chord_durations: List[float] = field(default_factory=list)

    # Functional-harmony ratios
    func_ratios: Dict[str, float] = field(default_factory=dict)

    # Bigrams
    bigrams: List[Tuple[str, str]] = field(default_factory=list)
    bigram_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Cadences detected
    cadences: List[Dict[str, Any]] = field(default_factory=list)

    # Modulations
    modulations: List[Dict[str, Any]] = field(default_factory=list)

    # musif features (if available)
    musif_features: Dict[str, float] = field(default_factory=dict)


class SemanticAnalyzer:
    """
    Semantic harmonic analysis of MIDI / MusicXML files.

    Uses music21 for key analysis and Roman-numeral chord labelling,
    and optionally delegates to **musif** for higher-level harmonic
    descriptors.

    Parameters
    ----------
    use_musif : bool
        If True and musif is installed, also run musif extraction.
    key_window : int
        Number of measures to use for windowed key analysis
        (for modulation detection).
    key_confidence : float
        Minimum correlation for Krumhansl-Schmuckler key finding.
    """

    def __init__(
        self,
        use_musif: bool = True,
        key_window: int = 6,
        key_confidence: float = 0.80,
    ) -> None:
        cfg_musif = CFG.get("features", {}).get("musif", {})
        self.use_musif = use_musif and cfg_musif.get("extract_harmony", True)
        self.key_window = key_window
        self.key_confidence = key_confidence

        # Check musif availability
        self._musif_available = False
        if self.use_musif:
            try:
                import musif  # noqa: F401
                self._musif_available = True
            except ImportError:
                warnings.warn(
                    "musif is not installed — semantic extraction will use music21 only. "
                    "Install via: pip install musif"
                )

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry: file → SemanticReport
    # ─────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        source: str | Path,
        verbose: bool = True,
    ) -> SemanticReport:
        """
        Perform semantic harmonic analysis on a MIDI / MusicXML file.

        Parameters
        ----------
        source : str or Path
            Path to a MIDI or MusicXML file.
        verbose : bool
            Print progress.

        Returns
        -------
        SemanticReport
        """
        from music21 import converter, key as m21key, roman

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if verbose:
            print(f"  🎵 Analysing: {source.name}")

        report = SemanticReport(file_path=str(source))

        # Parse with music21
        try:
            score = converter.parse(str(source))
        except Exception as exc:
            warnings.warn(f"music21 parse failed for {source}: {exc}")
            return report

        # ── Global key analysis ───────────────────────────────────────
        try:
            analysis = score.analyze("key")
            report.global_key = analysis.tonic.name
            report.global_mode = analysis.mode
            report.key_confidence = analysis.correlationCoefficient
        except Exception:
            report.global_key = "C"
            report.global_mode = "major"
            report.key_confidence = 0.0

        # ── Extract chords and Roman numerals ─────────────────────────
        try:
            chords_and_rests = score.chordify()
            global_key_obj = m21key.Key(report.global_key, report.global_mode)

            for chord_obj in chords_and_rests.recurse().getElementsByClass("Chord"):
                try:
                    rn = roman.romanNumeralFromChord(chord_obj, global_key_obj)
                    report.roman_numerals.append(rn.romanNumeralAlone)
                    report.chord_labels.append(rn.figure)
                    report.chord_durations.append(float(chord_obj.duration.quarterLength))
                except Exception:
                    # Fallback: use pitch name string
                    label = " ".join(p.name for p in chord_obj.pitches)
                    report.chord_labels.append(label)
                    report.chord_durations.append(float(chord_obj.duration.quarterLength))
        except Exception as exc:
            if verbose:
                print(f"    ⚠ Chord extraction failed: {exc}")

        # ── Functional-harmonic ratios ────────────────────────────────
        if report.roman_numerals:
            func_counter: Counter = Counter()
            for rn in report.roman_numerals:
                base = rn.rstrip("0123456789+o/")
                func = _ROMAN_FUNCTIONS.get(base, "other")
                func_counter[func] += 1
            total = sum(func_counter.values())
            report.func_ratios = {
                k: round(v / total, 4) for k, v in func_counter.items()
            }

        # ── Bigrams ───────────────────────────────────────────────────
        if len(report.roman_numerals) >= 2:
            report.bigrams = list(zip(report.roman_numerals[:-1], report.roman_numerals[1:]))
            report.bigram_counts = dict(Counter(report.bigrams))

        # ── Cadence detection ─────────────────────────────────────────
        if len(report.roman_numerals) >= 2:
            for i in range(len(report.roman_numerals) - 1):
                pair = (report.roman_numerals[i], report.roman_numerals[i + 1])
                base_pair = (
                    pair[0].rstrip("0123456789+o/"),
                    pair[1].rstrip("0123456789+o/"),
                )
                if base_pair in _CADENCE_PATTERNS:
                    report.cadences.append({
                        "type": _CADENCE_PATTERNS[base_pair],
                        "position": i,
                        "chords": pair,
                    })

        # ── Modulation detection (windowed key analysis) ──────────────
        try:
            measures = list(score.parts[0].getElementsByClass("Measure")) if score.parts else []
            n_measures = len(measures)
            if n_measures > self.key_window:
                prev_key = report.global_key
                for start in range(0, n_measures - self.key_window + 1, self.key_window // 2):
                    window = score.measures(start + 1, start + self.key_window)
                    try:
                        k = window.analyze("key")
                        if (k.tonic.name != prev_key and
                                k.correlationCoefficient >= self.key_confidence):
                            report.modulations.append({
                                "from_key": prev_key,
                                "to_key": k.tonic.name,
                                "mode": k.mode,
                                "measure": start,
                                "confidence": round(k.correlationCoefficient, 4),
                            })
                            prev_key = k.tonic.name
                    except Exception:
                        pass
        except Exception:
            pass

        # ── musif features (if available) ─────────────────────────────
        if self._musif_available:
            try:
                report.musif_features = self._extract_musif(source)
            except Exception as exc:
                if verbose:
                    print(f"    ⚠ musif extraction failed: {exc}")

        return report

    # ─────────────────────────────────────────────────────────────────────
    #  Flat feature vector from report
    # ─────────────────────────────────────────────────────────────────────

    def extract_features(self, report: SemanticReport) -> Dict[str, Any]:
        """
        Flatten a SemanticReport into a dict of scalar / fixed-size features
        suitable for DataFrame construction or tensor conversion.
        """
        feats: Dict[str, Any] = OrderedDict()

        # Key & mode
        feats["sem_global_key"] = report.global_key
        feats["sem_global_mode"] = report.global_mode
        feats["sem_key_confidence"] = report.key_confidence

        # Chord counts
        feats["sem_num_chords"] = len(report.roman_numerals)
        feats["sem_unique_chords"] = len(set(report.roman_numerals)) if report.roman_numerals else 0
        feats["sem_unique_chord_ratio"] = (
            feats["sem_unique_chords"] / max(1, feats["sem_num_chords"])
        )

        # Functional ratios
        for func in ("T", "D", "S", "PD", "other"):
            feats[f"sem_func_ratio_{func}"] = report.func_ratios.get(func, 0.0)

        # Modulations
        feats["sem_num_modulations"] = len(report.modulations)

        # Cadences
        feats["sem_num_cadences"] = len(report.cadences)
        cadence_types = Counter(c["type"] for c in report.cadences)
        for ct in ("authentic", "plagal", "deceptive", "half"):
            feats[f"sem_cadence_{ct}"] = cadence_types.get(ct, 0)

        # Duration statistics
        if report.chord_durations:
            durs = np.array(report.chord_durations)
            feats["sem_dur_mean"] = float(np.mean(durs))
            feats["sem_dur_std"] = float(np.std(durs))
            feats["sem_dur_max"] = float(np.max(durs))
        else:
            feats["sem_dur_mean"] = 0.0
            feats["sem_dur_std"] = 0.0
            feats["sem_dur_max"] = 0.0

        # Top bigrams (as string)
        if report.bigram_counts:
            top5 = sorted(report.bigram_counts.items(), key=lambda x: -x[1])[:5]
            feats["sem_top_bigrams"] = ";".join(f"{a}->{b}" for (a, b), _ in top5)
        else:
            feats["sem_top_bigrams"] = ""

        # Merge musif features
        for k, v in report.musif_features.items():
            feats[f"musif_{k}"] = v

        return feats

    # ─────────────────────────────────────────────────────────────────────
    #  musif helper
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_musif(source: Path) -> Dict[str, float]:
        """
        Extract features using musif library.

        musif provides higher-level harmonic descriptors including key
        profiles, interval-class vectors, and voice-leading metrics.
        """
        try:
            from musif.extract import FeaturesExtractor
        except ImportError:
            return {}

        try:
            extractor = FeaturesExtractor(
                [str(source)],
                features="all",
            )
            df = extractor.extract()
            if df.empty:
                return {}
            # Return first row as dict, filtering to numeric columns
            row = df.select_dtypes(include="number").iloc[0]
            return row.to_dict()
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────────────────────
    #  Batch extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        midi_paths: Sequence[str | Path],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run semantic analysis on multiple files and return a DataFrame.

        Each row = one file; columns = flat semantic features.
        """
        import pandas as pd

        rows = []
        n = len(midi_paths)
        for i, p in enumerate(midi_paths):
            try:
                report = self.analyze(p, verbose=False)
                feats = self.extract_features(report)
                feats["_file"] = Path(p).stem
                rows.append(feats)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ [{i+1}/{n}] {Path(p).name}: {exc}")

            if verbose and n >= 10 and (i + 1) % max(1, n // 10) == 0:
                print(f"  Progress: {i+1}/{n} files …")

        df = pd.DataFrame(rows)
        if "_file" in df.columns:
            df = df.set_index("_file")
        if verbose:
            print(f"  ✓ Extracted {len(df.columns)} semantic features × {len(df)} files")
        return df
