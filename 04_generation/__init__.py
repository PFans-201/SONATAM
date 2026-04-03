"""
04_generation — Audio → MIDI / MusicXML output utilities.

Modules
-------
midi_writer     : chord-progression → MIDI file (music21)
musicxml_writer : chord-progression → MusicXML file (music21)
"""

from .midi_writer     import write_chord_midi, progression_to_score
from .musicxml_writer import write_musicxml, score_to_musicxml

__all__ = [
    "write_chord_midi",
    "progression_to_score",
    "write_musicxml",
    "score_to_musicxml",
]
