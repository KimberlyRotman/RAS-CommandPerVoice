# pip install faster-whisper sounddevice numpy
import os
import queue
import re
import threading
import time
import tkinter as tk
import unicodedata

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PIL import Image, ImageTk

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
MAX_BUFFER_SECONDS = CHUNK_SECONDS * 3  # descarta áudio mais antigo se o processamento atrasar
MODEL_SIZE = "small"
LANGUAGE = "pt"
CPU_THREADS = os.cpu_count() or 4
CAMINHO_IMAGEM_OUVINDO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ouvindo.jpg")

DIRECOES = {"frente": "Frente", "tras": "Trás", "direita": "Direita", "esquerda": "Esquerda"}
LADOS = {"direito": "Direito", "esquerdo": "Esquerdo"}


def normalizar(texto: str) -> str:
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def identificar_comando(texto_normalizado: str):
    m = re.search(r"vai (?:pra|para) a? ?(frente|tras|direita|esquerda)", texto_normalizado)
    if m:
        return f"Vai pra {DIRECOES[m.group(1)]}"
    if re.search(r"\bme segue\b", texto_normalizado):
        return "Me Segue"
    if re.search(r"meia volta", texto_normalizado):
        return "Dê Meia Volta"
    if re.search(r"volta completa", texto_normalizado):
        return "Dê Volta Completa"
    m = re.search(r"fica do meu lado (direito|esquerdo)", texto_normalizado)
    if m:
        return f"Fica do Meu Lado {LADOS[m.group(1)]}"
    return None


class OuvinteDeAudio:
    """Grava o microfone continuamente e transcreve em blocos, publicando o texto numa fila."""

    def __init__(self, model, sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.text_queue = queue.Queue()
        self._buffer = []
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._stream = None
        self._worker = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        with self._lock:
            self._buffer.append(indata.copy())

    def start(self):
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _process_loop(self):
        while self._running.is_set():
            time.sleep(self.chunk_seconds)
            with self._lock:
                if not self._buffer:
                    continue
                audio = np.concatenate(self._buffer, axis=0).flatten()
                self._buffer.clear()
            if audio.size == 0:
                continue
            max_amostras = int(MAX_BUFFER_SECONDS * self.sample_rate)
            if audio.size > max_amostras:
                audio = audio[-max_amostras:]  # descarta o áudio mais antigo para não acumular atraso
            duracao_audio = audio.size / self.sample_rate
            inicio_processamento = time.time()
            segments, _ = self.model.transcribe(
                audio, language=LANGUAGE, beam_size=1, vad_filter=True
            )
            texto = "".join(segment.text for segment in segments).strip()
            duracao_processamento = time.time() - inicio_processamento
            self.text_queue.put((texto, duracao_processamento, duracao_audio))


OPCOES = [
    "Me Segue",
    "Vai pra (Frente/Trás/Direita/Esquerda)",
    "Dê Meia Volta",
    "Dê Volta Completa",
    "Fica do Meu Lado (Direito/Esquerdo)",
]


class AssistenteApp:
    IDLE, ACORDADO, COMANDO, AGRADECIMENTO = "idle", "acordado", "comando", "agradecimento"

    def __init__(self, root: tk.Tk, ouvinte: OuvinteDeAudio):
        self.root = root
        self.ouvinte = ouvinte
        self.state = self.IDLE

        root.title("Assistente José")
        root.geometry("700x500")
        root.configure(bg="black")

        self._live_words = []
        self.top_transcript = tk.Label(
            root, text="", font=("Consolas", 12), bg="black", fg="gray", wraplength=680, justify="left", anchor="w"
        )
        self.top_transcript.pack(side="top", fill="x", padx=10, pady=(5, 0))

        self.bottom_frame = tk.Frame(root, bg="black")
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.options_label = tk.Label(
            self.bottom_frame,
            text="\n".join(OPCOES),
            font=("Consolas", 10),
            bg="black",
            fg="lightgray",
            justify="left",
            anchor="sw",
        )
        self.options_label.pack(side="left", anchor="sw")
        self.speed_label = tk.Label(
            self.bottom_frame,
            text="",
            font=("Consolas", 10),
            bg="black",
            fg="cyan",
            justify="right",
            anchor="se",
        )
        self.speed_label.pack(side="right", anchor="se")

        self.idle_frame = tk.Frame(root, bg="black")
        self.idle_label = tk.Label(
            self.idle_frame,
            text="Precisa de ajuda? Chame por José",
            font=("Consolas", 28, "bold"),
            bg="black",
            fg="white",
            justify="center",
        )
        self.idle_label.pack(expand=True, fill="x", padx=20, pady=(40, 20))
        self.idle_frame.bind("<Configure>", self._atualizar_wraplength_idle)

        self.canvas = tk.Canvas(root, width=700, height=500, bg="black", highlightthickness=0)
        self.canvas.bind("<Configure>", self._redesenhar_canvas_ouvindo)
        self._imagem_ouvindo_original = Image.open(CAMINHO_IMAGEM_OUVINDO)
        self._imagem_ouvindo_tk = None

        self.command_label = tk.Label(
            root, text="", font=("Consolas", 26, "bold"), bg="black", fg="lightgreen", wraplength=650
        )

        self.thanks_label = tk.Label(
            root, text="Ao seu dispor", font=("Consolas", 30, "bold"), bg="black", fg="white"
        )

        self._mostrar_idle()
        self._poll_queue()

    def _limpar_tela(self):
        for widget in (self.idle_frame, self.canvas, self.command_label, self.thanks_label):
            widget.pack_forget()

    def _mostrar_idle(self):
        self._limpar_tela()
        self.idle_frame.pack(fill="both", expand=True)

    def _mostrar_acordado(self):
        self._limpar_tela()
        self.canvas.pack(fill="both", expand=True)

    def _atualizar_wraplength_idle(self, event):
        self.idle_label.config(wraplength=max(event.width - 40, 100))

    def _redesenhar_canvas_ouvindo(self, event):
        largura, altura = event.width, event.height
        if largura <= 1 or altura <= 1:
            return
        self.canvas.delete("ouvindo")
        max_largura = int(largura * 0.8)
        max_altura = int(altura * 0.75)
        if max_largura <= 0 or max_altura <= 0:
            return
        imagem = self._imagem_ouvindo_original.copy()
        imagem.thumbnail((max_largura, max_altura), Image.Resampling.LANCZOS)  # mantém a proporção original
        self._imagem_ouvindo_tk = ImageTk.PhotoImage(imagem)
        self.canvas.create_image(largura / 2, altura * 0.4, image=self._imagem_ouvindo_tk, tags="ouvindo")
        self.canvas.create_text(
            largura / 2, altura * 0.85, text="estou ouvindo", fill="white", font=("Consolas", 22), tags="ouvindo"
        )

    def _mostrar_comando(self, texto: str):
        self._limpar_tela()
        self.command_label.config(text=texto)
        self.command_label.pack(fill="both", expand=True, padx=10, pady=10)

    def _mostrar_agradecimento(self):
        self._limpar_tela()
        self.thanks_label.pack(fill="both", expand=True)
        self.root.after(3000, self._voltar_idle_apos_agradecimento)

    def _voltar_idle_apos_agradecimento(self):
        if self.state == self.AGRADECIMENTO:
            self.state = self.IDLE
            self._mostrar_idle()

    def _atualizar_transcricao_topo(self, texto: str):
        for palavra in texto.split():
            if len(self._live_words) >= 7:
                self._live_words = []
            self._live_words.append(palavra)
        self.top_transcript.config(text=" ".join(self._live_words))

    def _poll_queue(self):
        try:
            while True:
                texto, duracao_processamento, duracao_audio = self.ouvinte.text_queue.get_nowait()
                self._atualizar_velocidade(duracao_processamento, duracao_audio)
                if texto:
                    self._processar_texto(texto)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _atualizar_velocidade(self, duracao_processamento: float, duracao_audio: float):
        rtf = duracao_processamento / duracao_audio if duracao_audio > 0 else 0
        self.speed_label.config(
            text=f"Processamento: {duracao_processamento:.2f}s | Áudio: {duracao_audio:.2f}s | RTF: {rtf:.2f}x"
        )

    def _processar_texto(self, texto: str):
        self._atualizar_transcricao_topo(texto)
        normalizado = normalizar(texto)

        if self.state == self.IDLE:
            if "jose" in normalizado:
                self.state = self.ACORDADO
                self._mostrar_acordado()
            return

        if self.state == self.ACORDADO:
            comando = identificar_comando(normalizado)
            if comando:
                self.state = self.COMANDO
                self._mostrar_comando(comando)
            return

        if self.state == self.COMANDO:
            if re.search(r"\bobrigad[oa]\b", normalizado):
                self.state = self.AGRADECIMENTO
                self._mostrar_agradecimento()
            return


def main():
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
    ouvinte = OuvinteDeAudio(model)
    ouvinte.start()

    root = tk.Tk()
    AssistenteApp(root, ouvinte)

    def ao_fechar():
        ouvinte.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", ao_fechar)
    root.mainloop()


if __name__ == "__main__":
    main()
