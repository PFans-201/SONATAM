"""
audio_transcriber.py
====================
Wrapper for Google's **MT3** (Music Transcription Transformer) to convert
raw audio (MP3 / WAV / FLAC) into a MIDI file.

MT3 is an optional dependency; install via::

    pip install -e ".[mt3]"
    # or: pip install git+https://github.com/magenta/mt3.git

This module provides a clean interface that:
1. Loads audio → resamples to 16 kHz mono
2. Runs MT3 inference → note events
3. Writes output to a temporary MIDI file for downstream extraction

Main class
----------
AudioTranscriber
    transcribe(audio_path) → Path   (MIDI file)
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Optional

from sonata.config.settings import CFG

__all__ = ["AudioTranscriber"]


class AudioTranscriber:
    """
    Transcribe audio to MIDI using MT3.

    Falls back to ``basic-pitch`` (Spotify) if MT3 is not installed.

    Parameters
    ----------
    model_name : str
        MT3 checkpoint name (default from config).
    sample_rate : int
        Target sample rate for the model (default: 16000).
    output_dir : str or Path, optional
        Directory to save transcribed MIDI files.
        ``None`` → use a temporary directory.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        sample_rate: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        cfg_mt3 = CFG.get("inference", {}).get("mt3", {})
        self.model_name = model_name or cfg_mt3.get("model_name", "mt3")
        self.sample_rate = sample_rate or cfg_mt3.get("sample_rate", 16000)
        self.output_dir = Path(output_dir) if output_dir else None

        # Detect available backend
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect which transcription backend is available."""
        try:
            import note_seq  # noqa: F401
            return "mt3"
        except ImportError:
            pass

        try:
            import basic_pitch  # noqa: F401
            return "basic_pitch"
        except ImportError:
            pass

        warnings.warn(
            "No audio transcription backend found. "
            "Install MT3 or basic-pitch: pip install basic-pitch"
        )
        return "none"

    def transcribe(
        self,
        audio_path: str | Path,
        output_path: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> Path:
        """
        Transcribe an audio file to MIDI.

        Parameters
        ----------
        audio_path : str or Path
            Input audio file (MP3, WAV, FLAC, etc.).
        output_path : str or Path, optional
            Where to save the MIDI file.  ``None`` → auto-generated.
        verbose : bool
            Print progress.

        Returns
        -------
        Path
            Path to the output MIDI file.

        Raises
        ------
        RuntimeError
            If no transcription backend is available.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if output_path is None:
            out_dir = self.output_dir or Path(tempfile.mkdtemp(prefix="mt3_"))
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{audio_path.stem}.mid"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"  🎤 Transcribing: {audio_path.name}  [{self._backend}]")

        if self._backend == "mt3":
            self._transcribe_mt3(audio_path, output_path)
        elif self._backend == "basic_pitch":
            self._transcribe_basic_pitch(audio_path, output_path)
        else:
            raise RuntimeError(
                "No transcription backend available. Install MT3 or basic-pitch."
            )

        if verbose:
            print(f"  ✓ MIDI saved → {output_path}")

        return output_path

    # ─────────────────────────────────────────────────────────────────────
    #  MT3 backend
    # ─────────────────────────────────────────────────────────────────────

    def _transcribe_mt3(self, audio_path: Path, output_path: Path) -> None:
        """Transcribe using Google MT3."""
        import librosa
        import note_seq

        # Load and resample audio
        audio, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        # MT3 inference (simplified — actual MT3 requires t5x/jax setup)
        # This is a placeholder for the full MT3 pipeline
        try:
            from mt3 import inference as mt3_infer

            sequence = mt3_infer.transcribe(audio, sample_rate=self.sample_rate)
            note_seq.sequence_proto_to_midi_file(sequence, str(output_path))
        except ImportError:
            raise RuntimeError(
                "MT3 inference module not found. "
                "Please install MT3: pip install git+https://github.com/magenta/mt3.git"
            )

    # ─────────────────────────────────────────────────────────────────────
    #  basic-pitch fallback
    # ─────────────────────────────────────────────────────────────────────

    def _transcribe_basic_pitch(self, audio_path: Path, output_path: Path) -> None:
        """Transcribe using Spotify's basic-pitch."""
        from basic_pitch.inference import predict_and_save

        out_dir = output_path.parent
        predict_and_save(
            [str(audio_path)],
            str(out_dir),
            save_midi=True,
            save_model_outputs=False,
            save_notes=False,
        )
        # basic-pitch saves as <stem>_basic_pitch.mid
        bp_output = out_dir / f"{audio_path.stem}_basic_pitch.mid"
        if bp_output.exists() and bp_output != output_path:
            bp_output.rename(output_path)
