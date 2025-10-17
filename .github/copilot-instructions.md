### Quick project summary

- Purpose: A small Streamlit app called "Unstructured Data Analysis" (`run.py`) that provides image, audio, and text workflows. Key libs: Streamlit, DeepFace, rembg, gTTS, SpeechRecognition, pydub, PIL, numpy.
- Structure: single-entry script `run.py` at repo root. A Python virtual environment is included under `streamlit_env/`.

### Architecture & data flows (what an AI coder should know)

- Single-process Streamlit web app. UI is organized into three tabs: Image Analysis, Audio Analysis, Text Analysis. See `run.py` for the exact component layout.
- Image flow: uploaded file -> PIL.Image -> numpy array -> DeepFace for detect/analyze -> rembg for background removal. DeepFace functions expect numpy arrays or file paths.
- Audio flow: text->gTTS->saved to `output.mp3`; uploaded audio -> pydub.AudioSegment -> exported to WAV in-memory -> SpeechRecognition's `sr.AudioFile` for transcription.

### Developer workflows & commands

- Activate the included venv (Windows cmd):

  - streamlit_env\Scripts\activate.bat

- Run the app:

  - after activation: `python run.py` or `streamlit run run.py` (preferred for Streamlit behavior)

- Dependencies are already vendored in `streamlit_env/Lib/site-packages`. If you recreate the environment, install packages with `pip install streamlit deepface rembg gTTS SpeechRecognition pydub pillow numpy`.

### Project-specific patterns & gotchas

- Single-file app: changes go into `run.py`. Prefer adding small helper modules if features grow.
- File I/O: gTTS writes `output.mp3` to the repo root. Consider switching to in-memory BytesIO to avoid filesystem side-effects when adding tests.
- Audio handling: uploaded audio is converted via pydub to WAV and passed to SpeechRecognition via `BytesIO`. Keep conversions in-memory where possible.
- DeepFace: `analyze` and `detectFace` calls in `run.py` expect images as numpy arrays and may raise exceptions when `enforce_detection=True`. Wrap calls in try/except (already done) and prefer `enforce_detection=False` when adding experiments.
- rembg: `remove` returns a PIL-compatible object; current code calls it directly on a PIL.Image.

### Integration points & external services

- Google Speech API via SpeechRecognition's `recognize_google` (network required). Expect transient network errors (handled in code).
- No other remote APIs by default, but DeepFace may download models on first run.

### Examples (copy-pasteable snippets from repo)

- Transcribe uploaded audio (core flow excerpt):

  - read file -> AudioSegment.from_file -> export to wav BytesIO -> `with sr.AudioFile(wav_io) as source: audio = r.record(source)` -> `r.recognize_google(audio)`

- Detect face:

  - `DeepFace.detectFace(img_array, enforce_detection=True)`

### Files to reference when editing or extending

- `run.py` — main app and all UI flows
- `streamlit_env/` — vendored virtualenv; use only as a hint of installed packages

### Testing & safety notes for AI agents

- Don't assume tests exist. Make minimal, local changes and run `streamlit run run.py` to verify UI behavior.
- Avoid adding secrets or external keys into files. The app uses public APIs (Google via SpeechRecognition) and local models.

If any of this is unclear or you'd like more detail (e.g., a safe unit-test harness or converting gTTS to in-memory streaming), tell me which area to expand and I will iterate.