# pip install faster-whisper sounddevice numpy pillow pygame
import ctypes
import glob
import os
import sys
import queue
import random
import re
import threading
import time
import tkinter as tk
import unicodedata

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import av
import numpy as np
import pygame
import sounddevice as sd
from faster_whisper import WhisperModel
from PIL import Image, ImageTk

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
MAX_BUFFER_SECONDS = CHUNK_SECONDS * 3  # descarta áudio mais antigo se o processamento atrasar
MODEL_SIZE = "small"
LANGUAGE = "pt"
CPU_THREADS = os.cpu_count() or 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_ROSTOS = os.path.join(BASE_DIR, "rostojose")
PASTA_HISTORIAS = os.path.join(BASE_DIR, "historinhas")
IMG_ACORDADO = os.path.join(PASTA_ROSTOS, "joseacordado.png")
IMG_OUVINDO = os.path.join(PASTA_ROSTOS, "joseouvindo.png")
IMG_HISTORIA = os.path.join(PASTA_ROSTOS, "josehistoria.png")
IMG_FECHADO = os.path.join(PASTA_ROSTOS, "Josefechado.png")

LARGURA_TELA, ALTURA_TELA = 1920, 1080
COR_FUNDO = "#FFFFFF"
COR_PRINCIPAL = "#1800AD"
COR_TITULO_HISTORIA = "#FF751F"

# Posição do olho esquerdo no joseacordado.png (imagem em 0,0) e destino ao ouvir "José"
OLHO_ESQUERDO_INICIAL_X = 571.4
OLHO_ESQUERDO_DESTINO_X = 108.0
DURACAO_MOVIMENTO_MS = 300
INTERVALO_ANIMACAO_MS = 16

# Barra de progresso da história (coordenadas do canto superior esquerdo da bolinha)
TAMANHO_BOLINHA = 65
BOLINHA_INICIO = (164.6, 908.8)
BOLINHA_FIM = (1617.0, 908.8)
ESPESSURA_LINHA = 15
INTERVALO_PROGRESSO_MS = 100

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
        proxima_leitura = time.monotonic() + self.chunk_seconds
        while self._running.is_set():
            # Desconta o tempo gasto transcrevendo, senão cada bloco fica maior e mais lento que o anterior
            time.sleep(max(0.0, proxima_leitura - time.monotonic()))
            proxima_leitura = time.monotonic() + self.chunk_seconds
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


def duracao_audio_segundos(caminho: str) -> float:
    with av.open(caminho) as container:
        return container.duration / av.time_base if container.duration else 0.0


class PlayerDeHistorias:
    """Toca uma lista sorteada de histórias (mp3) com pygame."""

    def __init__(self, pasta: str):
        self.arquivos = sorted(glob.glob(os.path.join(pasta, "*.mp3")))
        self.playlist = []
        self.indice = -1
        self.duracao = 0.0
        self.tocando = False
        pygame.mixer.init()

    def nova_playlist(self) -> bool:
        self.playlist = random.sample(self.arquivos, len(self.arquivos))
        self.indice = -1
        return self.ir_para(0)

    def ir_para(self, indice: int) -> bool:
        if not 0 <= indice < len(self.playlist):
            self.parar()
            return False
        self.indice = indice
        caminho = self.playlist[indice]
        self.duracao = duracao_audio_segundos(caminho)
        pygame.mixer.music.load(caminho)
        pygame.mixer.music.play()
        self.tocando = True
        return True

    def proxima(self) -> bool:
        return self.ir_para(self.indice + 1)

    def anterior(self) -> bool:
        return self.ir_para(self.indice - 1)

    def parar(self):
        self.tocando = False
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

    def terminou(self) -> bool:
        return self.tocando and not pygame.mixer.music.get_busy()

    def progresso(self) -> float:
        if self.duracao <= 0:
            return 0.0
        return min(max(pygame.mixer.music.get_pos() / 1000 / self.duracao, 0.0), 1.0)

    def titulo(self) -> str:
        return os.path.splitext(os.path.basename(self.playlist[self.indice]))[0]


class AssistenteApp:
    ACORDADO, TRANSICAO, OUVINDO, COMANDO, AGRADECIMENTO, HISTORIA = (
        "acordado", "transicao", "ouvindo", "comando", "agradecimento", "historia"
    )

    def __init__(self, root: tk.Tk, ouvinte: OuvinteDeAudio):
        self.root = root
        self.ouvinte = ouvinte
        self.state = self.ACORDADO
        self.player = PlayerDeHistorias(PASTA_HISTORIAS)
        self._geracao = 0  # invalida callbacks agendados de um estado anterior

        root.title("Assistente José")
        root.configure(bg=COR_FUNDO)
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))

        # As imagens são 1920x1080; só reduz se a tela real for menor, mantendo a proporção
        self.escala = min(root.winfo_screenwidth() / LARGURA_TELA, root.winfo_screenheight() / ALTURA_TELA, 1.0)
        self.largura = round(LARGURA_TELA * self.escala)
        self.altura = round(ALTURA_TELA * self.escala)

        self.canvas = tk.Canvas(
            root, width=self.largura, height=self.altura, bg=COR_FUNDO, highlightthickness=0
        )
        self.canvas.pack(expand=True)

        self._img_acordado = self._carregar_imagem(IMG_ACORDADO)
        self._img_ouvindo = self._carregar_imagem(IMG_OUVINDO)
        self._img_historia = self._carregar_imagem(IMG_HISTORIA)
        self._img_fechado = self._carregar_imagem(IMG_FECHADO)
        self.rosto = self.canvas.create_image(0, 0, image=self._img_acordado, anchor="nw")

        self._live_words = []
        self.transcricao = self.canvas.create_text(
            10, 10, text="", anchor="nw", fill="gray", font=("Consolas", 12)
        )
        self.velocidade = self.canvas.create_text(
            self.largura - 10, self.altura - 10, text="", anchor="se", fill="gray", font=("Consolas", 10)
        )

        self._mostrar_acordado()
        self._poll_queue()

    def _s(self, valor: float) -> float:
        return valor * self.escala

    def _fonte(self, pixels: int):
        return ("Arial", -max(round(pixels * self.escala), 1), "bold")

    def _carregar_imagem(self, caminho: str):
        imagem = Image.open(caminho)
        if self.escala < 1.0:
            imagem = imagem.resize((self.largura, self.altura), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(imagem)

    def _novo_estado(self, estado: str):
        self.state = estado
        self._geracao += 1
        self.canvas.delete("conteudo")
        self.canvas.itemconfig(self.rosto, state="normal")

    def _mostrar_rosto(self, imagem, x: float = 0):
        self.canvas.itemconfig(self.rosto, image=imagem)
        self.canvas.coords(self.rosto, x, 0)

    def _mostrar_acordado(self):
        self._novo_estado(self.ACORDADO)
        self._mostrar_rosto(self._img_acordado)

    def _iniciar_transicao_ouvindo(self):
        self._novo_estado(self.TRANSICAO)
        self._mostrar_rosto(self._img_acordado)
        self._animar_movimento(time.perf_counter(), self._geracao)

    def _animar_movimento(self, inicio: float, geracao: int):
        if geracao != self._geracao:
            return
        fracao = min((time.perf_counter() - inicio) * 1000 / DURACAO_MOVIMENTO_MS, 1.0)
        deslocamento = self._s(OLHO_ESQUERDO_DESTINO_X - OLHO_ESQUERDO_INICIAL_X) * fracao
        self.canvas.coords(self.rosto, deslocamento, 0)
        if fracao < 1.0:
            self.root.after(INTERVALO_ANIMACAO_MS, self._animar_movimento, inicio, geracao)
        else:
            self._mostrar_ouvindo()

    def _mostrar_ouvindo(self):
        self._novo_estado(self.OUVINDO)
        self._mostrar_rosto(self._img_ouvindo)

    def _mostrar_comando(self, texto: str):
        self._novo_estado(self.COMANDO)
        self._mostrar_rosto(self._img_fechado)
        # Texto na faixa vazia acima dos olhos (que começam em y≈477 na imagem)
        self.canvas.create_text(
            self.largura / 2, self._s(240), text=texto, fill=COR_PRINCIPAL,
            font=self._fonte(80), width=self.largura - self._s(200), justify="center", tags="conteudo"
        )

    def _mostrar_agradecimento(self):
        self._novo_estado(self.AGRADECIMENTO)
        self.canvas.itemconfig(self.rosto, state="hidden")
        self.canvas.create_text(
            self.largura / 2, self.altura / 2, text="Ao seu dispor", fill=COR_PRINCIPAL,
            font=self._fonte(93), tags="conteudo"
        )
        self.root.after(3000, self._voltar_acordado, self._geracao)

    def _voltar_acordado(self, geracao: int):
        if geracao == self._geracao:
            self._mostrar_acordado()

    def _iniciar_historias(self):
        if not self.player.nova_playlist():
            self._mostrar_acordado()
            return
        self._mostrar_historia_atual()

    def _mostrar_historia_atual(self):
        self._novo_estado(self.HISTORIA)
        self._mostrar_rosto(self._img_historia)
        raio = self._s(TAMANHO_BOLINHA / 2)
        (x0, y0), (x1, y1) = self._posicoes_bolinha()
        self.canvas.create_line(
            x0 + raio, y0 + raio, x1 + raio, y1 + raio,
            fill=COR_PRINCIPAL, width=self._s(ESPESSURA_LINHA), capstyle="round", tags="conteudo"
        )
        self.canvas.create_text(
            (x0 + x1) / 2 + raio, min(y0, y1) - self._s(20), text=self.player.titulo(), anchor="s",
            fill=COR_TITULO_HISTORIA, font=self._fonte(53), tags="conteudo"
        )
        self.bolinha = self.canvas.create_oval(
            x0, y0, x0 + 2 * raio, y0 + 2 * raio,
            fill=COR_PRINCIPAL, outline=COR_PRINCIPAL, tags="conteudo"
        )
        self._atualizar_progresso(self._geracao)

    def _posicoes_bolinha(self):
        return tuple((self._s(x), self._s(y)) for x, y in (BOLINHA_INICIO, BOLINHA_FIM))

    def _atualizar_progresso(self, geracao: int):
        if geracao != self._geracao:
            return
        if self.player.terminou():
            self._trocar_historia(self.player.proxima())
            return
        p = self.player.progresso()
        (x0, y0), (x1, y1) = self._posicoes_bolinha()
        x, y = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
        tamanho = self._s(TAMANHO_BOLINHA)
        self.canvas.coords(self.bolinha, x, y, x + tamanho, y + tamanho)
        self.root.after(INTERVALO_PROGRESSO_MS, self._atualizar_progresso, geracao)

    def _trocar_historia(self, tocando: bool):
        if tocando:
            self._mostrar_historia_atual()
        else:
            self._mostrar_acordado()

    def parar(self):
        self.player.parar()

    def _atualizar_transcricao_topo(self, texto: str):
        for palavra in texto.split():
            if len(self._live_words) >= 7:
                self._live_words = []
            self._live_words.append(palavra)
        self.canvas.itemconfig(self.transcricao, text=" ".join(self._live_words))

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
        self.canvas.itemconfig(
            self.velocidade,
            text=f"Processamento: {duracao_processamento:.2f}s | Áudio: {duracao_audio:.2f}s | RTF: {rtf:.2f}x",
        )

    def _processar_texto(self, texto: str):
        self._atualizar_transcricao_topo(texto)
        normalizado = normalizar(texto)

        if self.state == self.ACORDADO:
            if "jose" in normalizado:
                self._iniciar_transicao_ouvindo()
            return

        if self.state == self.OUVINDO:
            if re.search(r"\bhistori", normalizado):
                self._iniciar_historias()
                return
            comando = identificar_comando(normalizado)
            if comando:
                self._mostrar_comando(comando)
            return

        if self.state == self.COMANDO:
            if re.search(r"\bobrigad[oa]\b", normalizado):
                self._mostrar_agradecimento()
            return

        if self.state == self.HISTORIA:
            # "para" só vale no fim da frase ("Para!"), para não disparar com "voltou para casa"
            if re.search(r"\b(parar|pare)\b|\bpara\b\s*(?:[.!?,]|$)", normalizado):
                self.player.parar()
                self._mostrar_acordado()
            elif re.search(r"\bproxima\b", normalizado):
                self._trocar_historia(self.player.proxima())
            elif re.search(r"\bvolta(r)?\b", normalizado):
                self._trocar_historia(self.player.anterior())
            return


def main():
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
    ouvinte = OuvinteDeAudio(model)
    ouvinte.start()

    if sys.platform == "win32":
        # Evita que o Windows amplie a janela pela escala de exibição (125%, 150%...)
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = tk.Tk()
    app = AssistenteApp(root, ouvinte)

    def ao_fechar():
        ouvinte.stop()
        app.parar()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", ao_fechar)
    root.mainloop()


if __name__ == "__main__":
    main()
