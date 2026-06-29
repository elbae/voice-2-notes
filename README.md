# Voice2Notes Tray

A Windows system tray utility that records microphone audio and transcribes it locally into text files.

## Features
- **Tray-only Interface**: No GUI windows. Controlled entirely via right-click context menu (Play / Stop / Exit).
- **100% Offline Transcription**: Uses `faster-whisper` (configured for the `medium` model) for local inference. No data is sent to external servers.
- **Structured File Output**: Automatically creates folders named `YYYY-MM-DD-XX` containing the raw `registrazione.wav` and the output `trascrizione.txt`.
- **Resource Constraints**: Limits CPU usage to a fixed number of threads to prevent system slowdowns during transcription.
- **Recording Reminder**: Triggers a native system notification every X minutes of continuous recording.

