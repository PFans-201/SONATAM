"""
msd_reader.py
=============
Pure-h5py reader for Million Song Dataset (MSD) HDF5 files.

Works on Python 3.8+ without pytables.

Functions
---------
read_msd_metadata(h5_path)
    Read all useful scalar metadata + genre tags from a single-song .h5 file.
    Returns a flat dict ready to merge into a pandas DataFrame row.
"""

from __future__ import annotations

from typing import Dict

import h5py

__all__ = ["read_msd_metadata", "KEY_NAMES"]

# Chromatic pitch-class index → name (using sharps)
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def read_msd_metadata(h5_path: str) -> Dict:
    """
    Read all useful metadata from a single-song MSD ``.h5`` file.

    Extracts
    --------
    Identification
        artist_name, title, release, artist_id, song_id, track_id

    Audio analysis (Echo Nest)
        msd_key, msd_key_name, msd_key_conf
        msd_mode (0=minor / 1=major), msd_mode_name, msd_mode_conf
        msd_tempo, msd_time_sig, msd_duration, msd_loudness
        msd_danceability, msd_energy

    MusicBrainz
        year, mbtags (semicolon-separated string)

    Genre / tags (Echo Nest artist terms)
        artist_terms     — list of term strings
        artist_terms_freq — list of frequency floats
        primary_genre    — highest-frequency term
        top3_genres      — semicolon-joined top-3 terms

    Parameters
    ----------
    h5_path : str
        Absolute or relative path to the MSD HDF5 file (``TRXXXXX.h5``).

    Returns
    -------
    dict
        Flat mapping of field names → scalar values (or short lists).

    Raises
    ------
    OSError / KeyError
        If the file cannot be opened or expected datasets are missing.
    """
    info: Dict = {}

    with h5py.File(h5_path, "r") as h5:

        # ── /metadata/songs ──────────────────────────────────────────
        meta = h5["/metadata/songs"][0]
        info["artist_name"] = meta["artist_name"].decode("utf-8", errors="replace")
        info["title"]       = meta["title"].decode("utf-8", errors="replace")
        info["release"]     = meta["release"].decode("utf-8", errors="replace")
        info["artist_id"]   = meta["artist_id"].decode()
        info["song_id"]     = meta["song_id"].decode()

        # ── /analysis/songs ──────────────────────────────────────────
        ana = h5["/analysis/songs"][0]
        info["track_id"]       = ana["track_id"].decode()
        info["msd_key"]        = int(ana["key"])
        info["msd_key_name"]   = KEY_NAMES[int(ana["key"])]
        info["msd_key_conf"]   = float(ana["key_confidence"])
        info["msd_mode"]       = int(ana["mode"])           # 0 = minor, 1 = major
        info["msd_mode_name"]  = "major" if ana["mode"] == 1 else "minor"
        info["msd_mode_conf"]  = float(ana["mode_confidence"])
        info["msd_tempo"]      = float(ana["tempo"])
        info["msd_time_sig"]   = int(ana["time_signature"])
        info["msd_duration"]   = float(ana["duration"])
        info["msd_loudness"]   = float(ana["loudness"])
        info["msd_danceability"] = float(ana["danceability"])
        info["msd_energy"]     = float(ana["energy"])

        # ── /musicbrainz/songs ───────────────────────────────────────
        mb = h5["/musicbrainz/songs"][0]
        info["year"] = int(mb["year"])

        # ── Artist terms (genre tags) ─────────────────────────────────
        terms   = [t.decode("utf-8", errors="replace") for t in h5["/metadata/artist_terms"][:]]
        freqs   = h5["/metadata/artist_terms_freq"][:].tolist()
        info["artist_terms"]       = terms
        info["artist_terms_freq"]  = freqs

        if terms and freqs:
            pairs = sorted(zip(freqs, terms), reverse=True)
            info["primary_genre"] = pairs[0][1]
            info["top3_genres"]   = ";".join(t for _, t in pairs[:3])
        else:
            info["primary_genre"] = ""
            info["top3_genres"]   = ""

        # ── MusicBrainz tags ──────────────────────────────────────────
        mbtags = [t.decode("utf-8", errors="replace") for t in h5["/musicbrainz/artist_mbtags"][:]]
        info["mbtags"] = ";".join(t for t in mbtags if t)

    return info
