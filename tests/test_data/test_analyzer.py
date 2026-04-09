"""
test_analyzer.py
================
Unit tests for sonata.data.analyzer.MIDIHarmonicAnalyzer
"""

import pytest


def test_import():
    from sonata.data.analyzer import MIDIHarmonicAnalyzer
    assert MIDIHarmonicAnalyzer is not None


def test_instantiation():
    from sonata.data.analyzer import MIDIHarmonicAnalyzer
    analyzer = MIDIHarmonicAnalyzer(key_window=4, key_confidence=0.75)
    assert analyzer.key_window == 4
    assert analyzer.key_confidence == 0.75


def test_quality_to_harte_mapping():
    from sonata.data.analyzer import MIDIHarmonicAnalyzer
    assert MIDIHarmonicAnalyzer.QUALITY_TO_HARTE["major"] == "maj"
    assert MIDIHarmonicAnalyzer.QUALITY_TO_HARTE["minor"] == "min"
