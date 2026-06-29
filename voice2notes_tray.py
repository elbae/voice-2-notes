import os
import re
import sys
import threading
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem


APP_NAME = "Voice2Notes Tray"
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
REMINDER_SECONDS = 5 * 60
DEFAULT_MODEL_NAME = "medium"
TRANSCRIPTION_CPU_THREADS = 8
INITIAL_PROMPT = (
    "Questo è un audio in italiano che contiene termini tecnici di informatica in inglese e spagnolo."
)


def app_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_directory() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return app_directory()


class Voice2NotesTrayApp:
    def __init__(self) -> None:
        self.base_dir = app_directory()
        self.bundle_dir = bundled_directory()

        self.icon = Icon(
            APP_NAME,
            self._generate_icon(),
            APP_NAME,
            menu=Menu(
                MenuItem(
                    "Play (Inizia Registrazione)",
                    self.on_play,
                    enabled=lambda item: not self.is_recording and not self.is_transcribing,
                ),
                MenuItem(
                    "Stop (Ferma e Trascrive)",
                    self.on_stop,
                    enabled=lambda item: self.is_recording,
                ),
                MenuItem("Esci", self.on_exit),
            ),
        )

        self.is_recording = False
        self.is_transcribing = False
        self.stop_recording_event = threading.Event()
        self.stop_notifier_event = threading.Event()
        self.state_lock = threading.Lock()

        self.recording_thread: threading.Thread | None = None
        self.notifier_thread: threading.Thread | None = None

        self.current_session_dir: Path | None = None
        self.current_wav_path: Path | None = None
        self.current_recording_started_at: datetime | None = None

    def run(self) -> None:
        self.icon.run()

    def _generate_icon(self) -> Image.Image:
        image = Image.new("RGBA", (64, 64), (18, 28, 42, 255))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((6, 6, 58, 58), radius=14, fill=(28, 88, 140, 255))
        draw.ellipse((22, 12, 42, 32), fill=(241, 196, 15, 255))
        draw.rounded_rectangle((24, 30, 40, 46), radius=7, fill=(241, 196, 15, 255))
        draw.rectangle((29, 46, 35, 54), fill=(241, 196, 15, 255))
        draw.rounded_rectangle((20, 52, 44, 56), radius=2, fill=(241, 196, 15, 255))
        return image

    def on_play(self, icon: Icon, item: MenuItem) -> None:
        del icon, item
        with self.state_lock:
            if self.is_recording or self.is_transcribing:
                return

            session_dir = self._next_session_directory()
            session_dir.mkdir(parents=True, exist_ok=False)
            wav_path = session_dir / "registrazione.wav"

            self.current_session_dir = session_dir
            self.current_wav_path = wav_path
            self.current_recording_started_at = datetime.now()
            self.stop_recording_event.clear()
            self.stop_notifier_event.clear()
            self.is_recording = True

            self.recording_thread = threading.Thread(
                target=self._record_audio_worker,
                args=(wav_path,),
                daemon=True,
                name="audio-recorder",
            )
            self.notifier_thread = threading.Thread(
                target=self._recording_reminder_worker,
                daemon=True,
                name="recording-reminder",
            )
            self.recording_thread.start()
            self.notifier_thread.start()

        self.icon.update_menu()
        self._show_notification("Registrazione avviata", f"Salvataggio in {session_dir.name}")

    def on_stop(self, icon: Icon, item: MenuItem) -> None:
        del icon, item
        with self.state_lock:
            if not self.is_recording or self.is_transcribing:
                return
            self.is_recording = False
            self.is_transcribing = True
            wav_path = self.current_wav_path
            session_dir = self.current_session_dir
            recording_started_at = self.current_recording_started_at

        self.stop_recording_event.set()
        self.stop_notifier_event.set()
        self.icon.update_menu()

        threading.Thread(
            target=self._finalize_and_transcribe_worker,
            args=(wav_path, session_dir, recording_started_at),
            daemon=True,
            name="transcriber",
        ).start()

    def on_exit(self, icon: Icon, item: MenuItem) -> None:
        del item
        with self.state_lock:
            if self.is_recording:
                self._show_notification("Voice2Notes Tray", "Ferma prima la registrazione.")
                return
            if self.is_transcribing:
                self._show_notification("Voice2Notes Tray", "Trascrizione in corso, attendi la fine.")
                return

        icon.stop()

    def _next_session_directory(self) -> Path:
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        for index in range(1, 1000):
            candidate = self.base_dir / f"{date_prefix}-{index:02d}"
            if not candidate.exists():
                return candidate
        raise RuntimeError("Impossibile trovare un numero progressivo libero.")

    def _record_audio_worker(self, wav_path: Path) -> None:
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(SAMPLE_RATE)

                def audio_callback(indata, frames, callback_time, status) -> None:
                    del frames, callback_time
                    if status:
                        print(status, file=sys.stderr)
                    if self.stop_recording_event.is_set():
                        raise sd.CallbackStop()
                    # Stream the PCM chunks directly to disk to keep memory usage flat.
                    wf.writeframesraw(indata.tobytes())
                    wf._file.flush()

                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    callback=audio_callback,
                ):
                    while not self.stop_recording_event.is_set():
                        time.sleep(0.1)
        except Exception as exc:
            self._write_error_file(wav_path.parent if wav_path.parent.exists() else self.base_dir, exc)
            self._show_notification("Errore registrazione", str(exc))
            with self.state_lock:
                self.is_recording = False
                self.is_transcribing = False
        finally:
            self.stop_notifier_event.set()
            self.icon.update_menu()

    def _recording_reminder_worker(self) -> None:
        while not self.stop_notifier_event.wait(REMINDER_SECONDS):
            with self.state_lock:
                if not self.is_recording:
                    break
                session_name = self.current_session_dir.name if self.current_session_dir else "sessione corrente"
            self._show_notification("Registrazione ancora attiva", f"Sto ancora registrando: {session_name}")

    def _finalize_and_transcribe_worker(
        self,
        wav_path: Path | None,
        session_dir: Path | None,
        recording_started_at: datetime | None,
    ) -> None:
        try:
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=15)

            if not wav_path or not session_dir or not wav_path.exists():
                raise RuntimeError("File audio non trovato dopo lo stop.")

            self._show_notification("Trascrizione avviata", "Elaborazione offline in corso...")
            transcript = self._transcribe_file(wav_path, recording_started_at)
            transcript_path = session_dir / "trascrizione.txt"
            transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")
            self._show_notification("Trascrizione completata", f"Creato {transcript_path.name}")
        except Exception as exc:
            target_dir = session_dir if session_dir and session_dir.exists() else self.base_dir
            self._write_error_file(target_dir, exc)
            self._show_notification("Errore trascrizione", str(exc))
        finally:
            with self.state_lock:
                self.is_transcribing = False
                self.current_wav_path = None
                self.current_session_dir = None
                self.current_recording_started_at = None
            self.icon.update_menu()

    def _transcribe_file(self, wav_path: Path, recording_started_at: datetime | None) -> str:
        model_path = self._resolve_model_source()
        model = WhisperModel(
            str(model_path) if isinstance(model_path, Path) else model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=TRANSCRIPTION_CPU_THREADS,
        )
        segments, _info = model.transcribe(
            str(wav_path),
            language="it",
            beam_size=5,
            vad_filter=True,
            temperature=0.0, # nuovo
            initial_prompt=INITIAL_PROMPT,
        )
        transcript_lines = []
        fallback_started_at = recording_started_at or datetime.now()
        pending_separator = False

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            segment_started_at = fallback_started_at + timedelta(seconds=segment.start)
            timestamp = segment_started_at.strftime("%Y-%m-%d %H:%M:%S")
            formatted_lines, pending_separator = self._format_transcript_segment(
                text,
                timestamp,
                pending_separator,
            )
            transcript_lines.extend(formatted_lines)

        return "\n".join(transcript_lines)

    def _format_transcript_segment(
        self,
        text: str,
        timestamp: str,
        pending_separator: bool = False,
    ) -> tuple[list[str], bool]:
        lines: list[str] = []
        parts = re.split(r"(?i)\bnuovo blocco\b", text)

        for index, raw_part in enumerate(parts):
            part = raw_part.strip(" \t,;:.!?-")

            if pending_separator and part:
                lines.append("---")
                pending_separator = False

            if part:
                lines.append(self._apply_voice_macro_format(part, timestamp))

            if index < len(parts) - 1:
                pending_separator = True

        return lines, pending_separator

    def _apply_voice_macro_format(self, text: str, timestamp: str) -> str:
        note_match = re.match(r"(?i)^(nota|attenzione)\s*:\s*(.+)$", text)
        if note_match:
            label = note_match.group(1).upper()
            body = note_match.group(2).strip()
            emoji = "⚠️" if label == "ATTENZIONE" else "📝"
            return f"[{timestamp}] > {emoji} **{label}:** {body}"

        task_match = re.match(r"(?i)^(task|promemoria)\s*:\s*(.+)$", text)
        if task_match:
            body = task_match.group(2).strip()
            return f"[{timestamp}] - [ ] {body}"

        todo_match = re.match(r"(?i)^to\s*do\b\s*:?\s*(.+)$", text)
        if todo_match:
            body = todo_match.group(1).strip()
            return f"[{timestamp}] - [ ] {body}"

        return f"[{timestamp}] {text}"

    def _resolve_model_source(self) -> Path | str:
        override = os.environ.get("VOICE2NOTES_MODEL_DIR")
        candidates = []
        if override:
            candidates.append(Path(override))

        candidates.extend(
            [
                self.base_dir / "models" / "base",
                self.base_dir / "models" / "faster-whisper-base",
                self.bundle_dir / "models" / "base",
                self.bundle_dir / "models" / "faster-whisper-base",
                self.base_dir / "models" / "tiny",
                self.base_dir / "models" / "faster-whisper-tiny",
                self.bundle_dir / "models" / "tiny",
                self.bundle_dir / "models" / "faster-whisper-tiny",
            ]
        )

        for candidate in candidates:
            if (candidate / "config.json").exists():
                return candidate

        return DEFAULT_MODEL_NAME

    def _write_error_file(self, target_dir: Path, exc: Exception) -> None:
        error_path = target_dir / "errore.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_path.write_text(f"[{timestamp}] {type(exc).__name__}: {exc}\n", encoding="utf-8")

    def _show_notification(self, title: str, message: str) -> None:
        try:
            self.icon.notify(message, title)
        except Exception:
            pass


if __name__ == "__main__":
    Voice2NotesTrayApp().run()
