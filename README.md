# VeadoController - Standalone AI PNGTuber Controller

An advanced, all-in-one tool using MediaPipe AI to drive PNGTuber avatars (like VeadoTube or VTube Studio) through facial expressions and hand gestures.

## Features

- **All-in-One Executable**: AI models are embedded inside the `.exe`. No messy folders.
- **Advanced Tracking**: Real-time detection of smiles, winks, brow movements, and head tilts.
- **Thinking Mode**: Unique hand-to-chin gesture detection using Hand Landmarker AI.
- **Ghost Keys (F13-F20)**: Built-in support for non-physical keys to prevent keyboard conflicts.
- **Interactive Setup Wizard**: Easily bind your avatar's triggers with a 3-second delay timer.
- **Hold Timers**: Customizable minimum duration per emotion to prevent "flickering" animations.

## Downloads

### For Users (Standalone)

Download the latest version from the **[Releases](https://github.com///releases)** page.

1. Download `PNGTuber_Controller.exe`.
2. Run the application.
3. (Optional) A `config.json` will be created automatically in the same folder to save your settings.

### For Developers (Source Code)

If you are running the `main.py` script manually, you must download the Google MediaPipe models and place them in your project root:

- [Face Landmarker Model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- [Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

## Development Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the controller: `python main.py`.

## License

MIT License - Open for use and modification.
