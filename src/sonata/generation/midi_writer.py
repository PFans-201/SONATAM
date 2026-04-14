"""
midi_writer.py
==============
Convert a chord progression (list of Harte-label strings or Roman-numeral
strings) into a MIDI file using music21.

Functions
---------
progression_to_score(chords, key, tempo, duration_per_chord)
    Build a music21 Score from a chord list.

write_chord_midi(chords, output_path, key, tempo, duration_per_chord)
    Build the Score and write it to a ``.mid`` file.

Usage
-----
>>> from sonata.generation.midi_writer import write_chord_midi
>>> chords = ["C:maj", "A:min", "F:maj", "G:maj"]
>>> write_chord_midi(chords, "data/processed/my_progression.mid", key="C major", tempo=120)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import music21 as m21

__all__ = ["progression_to_score", "write_chord_midi"]

# ── Harte root → music21 pitch name ─────────────────────────────────────────
_HARTE_ROOT_MAP = {
    "Cb": "C-", "C": "C", "C#": "C#",
    "Db": "D-", "D": "D", "D#": "D#",
    "Eb": "E-", "E": "E", "E#": "E#",
    "Fb": "F-", "F": "F", "F#": "F#",
    "Gb": "G-", "G": "G", "G#": "G#",
    "Ab": "A-", "A": "A", "A#": "A#",
    "Bb": "B-", "B": "B", "B#": "B#",
}

# ── Harte quality → interval list above root ────────────────────────────────
_HARTE_QUALITY_MAP = {
    "maj":      [0, 4, 7],
    "min":      [0, 3, 7],
    "dim":      [0, 3, 6],
    "aug":      [0, 4, 8],
    "7":        [0, 4, 7, 10],
    "maj7":     [0, 4, 7, 11],
    "min7":     [0, 3, 7, 10],
    "hdim7":    [0, 3, 6, 10],
    "dim7":     [0, 3, 6, 9],
    "aug6":     [0, 4, 8, 9],
    "augmaj7":  [0, 4, 8, 11],
    "sus2":     [0, 2, 7],
    "sus4":     [0, 5, 7],
    "5":        [0, 7],       # power chord
}


def _harte_to_m21_chord(
    label: str,
    duration_ql: float = 1.0,
    octave: int = 4,
) -> m21.chord.Chord:
    """
    Convert a Harte-style label (e.g., ``"C:maj"``, ``"A#:min7"``) to a
    music21 Chord with the given quarter-length duration.
    """
    if ":" in label:
        root_str, quality = label.split(":", 1)
    else:
        root_str, quality = label, "maj"

    root_m21  = _HARTE_ROOT_MAP.get(root_str, root_str)
    semitones = _HARTE_QUALITY_MAP.get(quality, [0, 4, 7])   # default: major triad

    root_pitch = m21.pitch.Pitch(f"{root_m21}{octave}")
    pitches    = [root_pitch.transpose(s) for s in semitones]
    ch         = m21.chord.Chord(pitches)
    ch.quarterLength = duration_ql
    return ch


def progression_to_score(
    chords: List[str],
    key: str = "C major",
    tempo: int = 120,
    duration_per_chord: float = 1.0,
    time_signature: str = "4/4",
) -> m21.stream.Score:
    """
    Build a music21 Score from a list of Harte chord labels.

    Parameters
    ----------
    chords : list of str
        Harte-style labels, e.g. ``["C:maj", "A:min", "F:maj", "G:7"]``.
    key : str
        Key string understood by ``music21.key.Key()``, e.g. ``"C major"``.
    tempo : int
        BPM.
    duration_per_chord : float
        Quarter-length duration for each chord.
    time_signature : str
        Time signature string, e.g. ``"4/4"`` or ``"3/4"``.

    Returns
    -------
    music21.stream.Score
    """
    score = m21.stream.Score()
    part  = m21.stream.Part()

    # Header elements
    part.append(m21.tempo.MetronomeMark(number=tempo))
    part.append(m21.meter.TimeSignature(time_signature))

    # Key signature
    try:
        key_obj = m21.key.Key(key.split()[0], key.split()[1] if " " in key else "major")
        part.append(key_obj)
    except Exception:
        pass  # skip if key string can't be parsed

    for label in chords:
        ch = _harte_to_m21_chord(label, duration_ql=duration_per_chord)
        part.append(ch)

    score.append(part)
    return score


def write_chord_midi(
    chords: List[str],
    output_path: str,
    key: str = "C major",
    tempo: int = 120,
    duration_per_chord: float = 1.0,
    time_signature: str = "4/4",
    verbose: bool = True,
) -> str:
    """
    Write a chord progression to a MIDI file.

    Parameters
    ----------
    chords : list of str
        Harte-style chord labels.
    output_path : str
        Destination ``.mid`` file path.
    key, tempo, duration_per_chord, time_signature
        Passed to :func:`progression_to_score`.
    verbose : bool
        Print confirmation on success.

    Returns
    -------
    str — absolute path to the written file.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    score = progression_to_score(
        chords,
        key=key,
        tempo=tempo,
        duration_per_chord=duration_per_chord,
        time_signature=time_signature,
    )

    mf = m21.midi.translate.streamToMidiFile(score)
    mf.open(str(p), "wb")
    mf.write()
    mf.close()

    if verbose:
        print(f"  ✓ MIDI written: {p}  ({len(chords)} chords, tempo={tempo})")

    return str(p.resolve())
