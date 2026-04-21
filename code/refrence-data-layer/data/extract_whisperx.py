#!/usr/bin/env python3
"""
Extract word-level subtitles from UDIVA-HHOI mosaic videos using WhisperX.

For each video in mosaics/<session>/<session>_L_mosaic.mp4, this script:
  1. (Optional) Enhances audio with dynamic range compression to boost quiet speakers.
  2. Transcribes audio with Whisper (via WhisperX).
  3. Aligns transcription to get word-level timestamps.
  4. Optionally runs speaker diarization (requires HF token).
  5. Saves results as JSON and word-level SRT.

Usage:
    python extract_whisperx.py [--model large-v3] [--output_dir whisperx_out]
    python extract_whisperx.py --enhance_audio       # boost quiet speakers
    python extract_whisperx.py --hf_token <TOKEN>    # enable diarization
    python extract_whisperx.py --force                # overwrite existing output

Requirements:
    pip install whisperx torch torchaudio
    ffmpeg (for --enhance_audio)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def enhance_audio(video_path: str | Path, output_wav: str | Path) -> None:
    """Apply dynamic range compression + loudness normalisation via ffmpeg.

    Pipeline:
      1. acompressor: heavy compression (ratio 6:1, low threshold -30 dB)
         → brings quiet speech closer to loud speech in level.
      2. loudnorm: EBU R128 loudness normalisation to -16 LUFS
         → ensures consistent absolute level for Whisper.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-af", (
            "acompressor=threshold=-30dB:ratio=6:attack=5:release=100:makeup=8dB,"
            "loudnorm=I=-16:TP=-1.5:LRA=7"
        ),
        "-ac", "1",        # mono (channels are identical anyway)
        "-ar", "16000",    # WhisperX expects 16 kHz
        "-sample_fmt", "s16",
        str(output_wav),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio enhancement failed:\n{result.stderr[-500:]}"
        )


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    seconds %= 60
    s = int(seconds)
    ms = int(round((seconds - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_word_srt(words: list[dict], path: Path) -> None:
    """Write word-level SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, w in enumerate(words, 1):
            start = w.get("start", 0.0)
            end = w.get("end", start + 0.1)
            text = w.get("word", "")
            speaker = w.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def write_segment_srt(segments: list[dict], path: Path) -> None:
    """Write segment-level SRT file (sentence-level with word timestamps in JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0.0)
            end = seg.get("end", start)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def process_video(
    video_path: Path,
    output_dir: Path,
    model,
    align_metadata: dict,
    align_model,
    device: str,
    diarize_pipeline=None,
    language: str | None = None,
    batch_size: int = 16,
    compute_type: str = "float16",
    do_enhance_audio: bool = False,
    force: bool = False,
) -> None:
    """Process a single video: transcribe → align → (diarize) → save."""
    import whisperx

    session = video_path.parent.name
    stem = video_path.stem  # e.g. "001080_L_mosaic"
    out_session = output_dir / session
    out_session.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    json_out = out_session / f"{stem}.json"
    if json_out.exists() and not force:
        print(f"  [skip] {session}/{stem} — already exists (use --force to overwrite)")
        return

    print(f"  [transcribe] {session}/{stem} ...", flush=True)

    # 1. Load audio (optionally with enhancement)
    tmp_wav = None
    if do_enhance_audio:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        print(f"    Enhancing audio (dynamic range compression) ...")
        enhance_audio(video_path, tmp_wav.name)
        audio = whisperx.load_audio(tmp_wav.name)
    else:
        audio = whisperx.load_audio(str(video_path))

    # 2. Transcribe
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_lang = result.get("language", language or "unknown")
    print(f"    Language: {detected_lang}")

    # 3. Align for word-level timestamps
    #    Load alignment model for the detected language if needed
    if detected_lang in align_metadata:
        a_model, a_meta = align_metadata[detected_lang]
    else:
        print(f"    Loading alignment model for '{detected_lang}' ...")
        a_model, a_meta = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        align_metadata[detected_lang] = (a_model, a_meta)

    result = whisperx.align(
        result["segments"], a_model, a_meta, audio, device,
        return_char_alignments=False,
    )

    # 4. Optional diarization
    if diarize_pipeline is not None:
        print("    Diarizing ...")
        diarize_segments = diarize_pipeline(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. Collect all words
    all_words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            all_words.append(w)

    # 6. Save outputs
    # Full JSON with segments + words
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "language": detected_lang,
                "segments": result.get("segments", []),
                "word_count": len(all_words),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Word-level SRT
    word_srt = out_session / f"{stem}_words.srt"
    write_word_srt(all_words, word_srt)

    # Segment-level SRT
    seg_srt = out_session / f"{stem}_segments.srt"
    write_segment_srt(result.get("segments", []), seg_srt)

    print(f"    → {len(result.get('segments', []))} segments, {len(all_words)} words")

    # Clean up temp file if used
    if tmp_wav is not None:
        try:
            os.unlink(tmp_wav.name)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Extract word-level subtitles from mosaic videos using WhisperX."
    )
    parser.add_argument(
        "--mosaics_dir",
        default="mosaics",
        help="Root directory of mosaic videos (default: mosaics/)",
    )
    parser.add_argument(
        "--output_dir",
        default="whisperx_out",
        help="Output directory (default: whisperx_out/)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. 'es', 'en'). Default: auto-detect.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16)",
    )
    parser.add_argument(
        "--compute_type",
        default="float16",
        help="Compute type for Whisper (default: float16)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token for speaker diarization (optional).",
    )
    parser.add_argument(
        "--enhance_audio",
        action="store_true",
        help="Apply dynamic range compression to boost quiet speakers (requires ffmpeg).",
    )
    parser.add_argument(
        "--vad_onset",
        type=float,
        default=None,
        help="VAD onset threshold (default: 0.5; lower = more sensitive to quiet speech, e.g. 0.2).",
    )
    parser.add_argument(
        "--no_speech_threshold",
        type=float,
        default=None,
        help="No-speech probability threshold (default: 0.6; higher = keep more borderline speech, e.g. 0.9).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Process only these sessions (default: all).",
    )
    args = parser.parse_args()

    # Import here so --help works without GPU
    import whisperx

    mosaics_root = Path(args.mosaics_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect videos
    videos = sorted(mosaics_root.rglob("*.mp4"))
    if args.sessions:
        sessions_set = set(args.sessions)
        videos = [v for v in videos if v.parent.name in sessions_set]

    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} video(s)")
    print(f"Model: {args.model}, device: {args.device}, compute: {args.compute_type}")
    if args.enhance_audio:
        print("Audio enhancement: ON (dynamic range compression)")
    if args.vad_onset is not None:
        print(f"VAD onset: {args.vad_onset} (default: 0.5)")
    if args.no_speech_threshold is not None:
        print(f"No-speech threshold: {args.no_speech_threshold} (default: 0.6)")
    print()

    # Build optional overrides
    vad_options = {}
    if args.vad_onset is not None:
        vad_options["vad_onset"] = args.vad_onset
    asr_options = {}
    if args.no_speech_threshold is not None:
        asr_options["no_speech_threshold"] = args.no_speech_threshold

    # Load Whisper model
    print("Loading Whisper model ...")
    model = whisperx.load_model(
        args.model, args.device, compute_type=args.compute_type,
        vad_options=vad_options or None,
        asr_options=asr_options or None,
    )

    # Alignment model cache (per language)
    align_metadata: dict = {}

    # Optional: diarization pipeline
    diarize_pipeline = None
    if args.hf_token:
        print("Loading diarization pipeline ...")
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=args.hf_token, device=args.device
        )

    # Process each video
    for video_path in videos:
        try:
            process_video(
                video_path=video_path,
                output_dir=output_root,
                model=model,
                align_metadata=align_metadata,
                align_model=None,
                device=args.device,
                diarize_pipeline=diarize_pipeline,
                language=args.language,
                batch_size=args.batch_size,
                compute_type=args.compute_type,
                do_enhance_audio=args.enhance_audio,
                force=args.force,
            )
        except Exception as e:
            print(f"  [ERROR] {video_path}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
