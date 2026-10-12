# RAS Voice Assistent

This repository contains one component of the RAS (Autonomous Service Robot) project. This component consists of a voice assistant that accepts commands and instructions to trigger specific functions:

- Follow me
- Move (forward, backward, right, left)
- Return
- Full turn
- Return to previous state
- Tell me a story

The assistant is called **José**. It listens to the microphone continuously, transcribes speech offline with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Portuguese) and reacts through a full-screen face interface.

# Installation

```bash
pip install faster-whisper sounddevice numpy pillow pygame
python jose.py
```

The window opens in full screen (1920x1080, white background). Press `Esc` to leave full screen. If the screen is smaller than 1920x1080, everything is scaled down proportionally.

# Repository structure

| Path | Content |
| --- | --- |
| `jose.py` | Voice assistant application |
| `rostojose/` | José's faces (1920x1080 PNG): `joseacordado`, `joseouvindo`, `Josefechado`, `josehistoria` |
| `historinhas/` | Story audio files (`.mp3`). The file name (without `.mp3`) is shown as the story title |

# How it works

1. **Awake** – On start, José shows `joseacordado.png`.
2. **Listening** – When he hears **"José"**, the face slides left (left eye from x 571.4 to x 108) in 0.3 s and switches to `joseouvindo.png`, waiting for an instruction.
3. **Instruction** – When a command is recognized, `Josefechado.png` is shown with the command text above it:
   - "Me segue"
   - "Vai pra frente / trás / direita / esquerda"
   - "Dê meia volta"
   - "Dê volta completa"
   - "Fica do meu lado direito / esquerdo"
4. **Thanks** – Saying **"Obrigado/Obrigada"** shows "Ao seu dispor" for 3 s and returns to the awake face.

## Story mode

Say **"História"** while José is listening. A new random order of all stories in `historinhas/` is drawn each time the mode is activated, and the stories play one after another from start to end.

The screen shows `josehistoria.png`, the story title in bold orange (`#FF751F`) and a progress bar: a 65x65 ball (`#1800AD`) moving along a 15 px line from (164.6, 908.8) to (1617, 908.8).

Voice commands during a story:

| Command | Action |
| --- | --- |
| "Para" / "Parar" / "Pare" | Stops the story and returns to the awake face |
| "Próxima" | Skips to the next story |
| "Volta" / "Voltar" | Goes back to the previous story |

If there is no next or previous story, José returns to the awake face.

> Note: the microphone also picks up the story audio, so a command word spoken in the narration may trigger that command.

# Authors
- Kimberly Rotman




