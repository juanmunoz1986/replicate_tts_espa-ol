import os
import sys
import torch
import torchaudio
from pathlib import Path
import tempfile

# Añadimos el directorio 'src' al path para poder importar f5_tts
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from replicate import Predictor, Input, Path as ReplicatePath

from f5_tts.model import CFM
from f5_tts.infer.utils_infer import (
    load_model,
    prepare_text,
    get_text_prompt_list,
    get_ref_prompt,
    prompt_map_to_prompt,
)
from vocos import Vocos

# --- Constantes del Modelo ---
# Modelo F5-TTS en Español
MODEL_CACHE = "f5-tts-spanish-cache"
MODEL_REPO = "jpgallegoar/F5-Spanish"
MODEL_FILENAME = "F5_es.pth"

# Vocoder
VOCODER_REPO = "charactr/vocos-mel-24khz"

class ReplicatePredictor(Predictor):
    def setup(self):
        """Carga el modelo y el vocoder en memoria. Se ejecuta una sola vez."""
        print("--- Iniciando el setup del predictor ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Cargando Vocoder desde {VOCODER_REPO}...")
        self.vocoder = Vocos.from_pretrained(VOCODER_REPO).to(self.device)
        
        print("Cargando modelo F5-TTS...")
        self.model = load_model("F5-TTS", os.path.join(MODEL_CACHE, MODEL_FILENAME), MODEL_REPO, self.device)
        self.model.eval()

        # Audio de referencia por defecto de alta calidad para clonación de voz
        # (Este audio se incluye en el repositorio para que siempre esté disponible)
        # Asegúrate de tener un archivo 'es_ref.wav' en la raíz.
        self.default_ref_audio = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
        print("--- Setup completado ---")

    def predict(
        self,
        text: str = Input(description="Texto en español para sintetizar."),
        ref_audio: ReplicatePath = Input(
            description="Opcional. Archivo de audio de referencia (WAV) para clonar un estilo de voz. Si no se proporciona, se usará una voz por defecto.",
            default=None,
        ),
        speed: float = Input(
            description="Velocidad de la síntesis de voz. 1.0 es normal, 1.5 es más rápido.",
            default=1.5,
            ge=0.5,
            le=2.0,
        ),
        temperature: float = Input(
            description="Temperatura de la generación. Valores más altos dan más variabilidad.",
            default=0.7,
            ge=0.0,
            le=1.0,
        )
    ) -> ReplicatePath:
        """Ejecuta una predicción de Texto a Voz."""
        print("--- Iniciando predicción ---")
        
        # Preparar texto y prompt
        prompt = {
            "main": {
                "text_prompt": prepare_text(text),
                "ref_prompt": None,
            }
        }

        # Usar audio de referencia si se proporciona, si no, usar el de por defecto
        ref_audio_path = str(ref_audio) if ref_audio else self.default_ref_audio
        print(f"Usando audio de referencia: {ref_audio_path}")
        
        prompt["main"]["ref_prompt"] = get_ref_prompt(
            self.model, ref_audio_path, self.device
        )

        prompt_list = prompt_map_to_prompt(prompt)

        print(f"Texto a generar: '{text}' con velocidad {speed}")
        
        # Generar espectrograma
        output_mel = self.model.forward_fair(
            prompt_list,
            steps=30,
            guidance=3,
            speed=speed,
            temperature=temperature,
        )
        
        print("Generando forma de onda con el vocoder...")
        # Convertir espectrograma a audio
        wav_gen = self.vocoder.decode(output_mel).cpu().squeeze(0)

        # Guardar el archivo de audio de salida
        output_path = Path(tempfile.mkdtemp()) / "output.wav"
        torchaudio.save(str(output_path), wav_gen, 24000)

        print(f"--- Predicción completada. Audio guardado en {output_path} ---")
        return ReplicatePath(output_path) 