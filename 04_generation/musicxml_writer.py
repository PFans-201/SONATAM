"""
musicxml_writer.py
==================
Convert a chord progression (list of Harte-label strings) or an existing
music21 Score to a MusicXML file — readable by Sibelius, Finale, MuseScore, etc.

Functions
---------
score_to_musicxml(score, output_path)
    Serialise an existing music21 Score to MusicXML.

write_musicxml(chords, output_path, key, tempo, duration_per_chord)
    Build a Score from chord labels and write it as MusicXML.

annotate_roman_numerals(score, key)
    Post-process a Score, adding Roman-numeral text annotations above each chord.

Dependencies
------------
    pip install music21
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import music21 as m21

from .midi_writer import progression_to_score

__all__ = ["score_to_musicxml", "write_musicxml", "annotate_roman_numerals"]


def score_to_musicxml(
    score: m21.stream.Score,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Write a music21 Score to a MusicXML file.

    Parameters
    ----------
    score : music21.stream.Score
    output_path : str
        Destination ``.musicxml`` or ``.xml`` file path.
    verbose : bool

    Returns
    -------
    str — absolute path to the written file.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    exporter = m21.musicxml.m21ToXml.ScoreExporter(score)
    xml_tree = exporter.parse()
    xml_tree.write(str(p), xml_declaration=True, encoding="utf-8")

    if verbose:
        print(f"  ✓ MusicXML written: {p}")
    return str(p.resolve())


def write_musicxml(
    chords: List[str],
    output_path: str,
    key: str = "C major",
    tempo: int = 120,
    duration_per_chord: float = 1.0,
    time_signature: str = "4/4",
    add_roman_annotations: bool = True,
    verbose: bool = True,
) -> str:
    """
    Build a chord-progression Score and export it as MusicXML.

    Parameters
    ----------
    chords : list of str
        Harte-style labels (e.g., ``["C:maj", "A:min", "F:maj", "G:7"]``).
    output_path : str
        Destination ``.musicxml`` file path.
    key, tempo, duration_per_chord, time_signature
        Passed to :func:`midi_writer.progression_to_score`.
    add_roman_annotations : bool
        If True, add Roman-numeral text expressions above each chord.
    verbose : bool

    Returns
    -------
    str — absolute path to the written file.
    """
    score = progression_to_score(
        chords,
        key=key,
        tempo=tempo,
        duration_per_chord=duration_per_chord,
        time_signature=time_signature,
    )

    if add_roman_annotations:
        score = annotate_roman_numerals(score, key)

    return score_to_musicxml(score, output_path, verbose=verbose)


def annotate_roman_numerals(
    score: m21.stream.Score,
    key: str = "C major",
) -> m21.stream.Score:
    """
    Add Roman-numeral text annotations (TextExpression) above each chord
    in the Score.

    Parameters
    ----------
    score : music21.stream.Score
    key : str
        Key context for Roman numeral computation.

    Returns
    -------
    The modified Score (in-place + returned).
    """
    try:
        key_parts = key.split()
        key_obj   = m21.key.Key(key_parts[0], key_parts[1] if len(key_parts) > 1 else "major")
    except Exception:
        return score

    for part in score.parts:
        for ch in part.recurse().getElementsByClass("Chord"):
            try:
                rn   = m21.roman.romanNumeralFromChord(ch, key_obj)
                expr = m21.expressions.TextExpression(rn.figure)
                expr.placement = "above"
                expr.style.fontSize = 9
                ch.activeSite.insert(ch.offset, expr)
            except Exception:
                pass  # skip if RN can't be computed

    return score
