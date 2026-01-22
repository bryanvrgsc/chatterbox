import random
import warnings
import os
import shutil
import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr

# === CONFIGURACIÓN Y CONSTANTES ===
MAX_CHARS = 10000
CHUNK_SIZE = 400  # Max 400 chars per chunk (model's native limit)

# --- OPTIMIZACIÓN: Procesamiento paralelo de chunks ---
ENABLE_PARALLEL_CHUNKS = False  # Cambiar a True para activar (experimental)
PARALLEL_WORKERS = 2  # Número de chunks a procesar en paralelo

# --- GESTIÓN DE MEMORIA GPU ---
USE_GPU_EMPTY_CACHE = False  # Cambiar a False para desactivar

# --- LIMPIEZA DE CACHÉ ---
AUTO_CLEAN_CACHE = True  # Cambiar a False para desactivar limpieza automática

# --- CARPETA DE SALIDA PERSISTENTE ---
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === OPTIMIZACIONES DE RENDIMIENTO ===
# Suprimir warnings no críticos para limpiar output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Detectar dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running on device: {DEVICE}")

# Optimizaciones específicas por dispositivo
if DEVICE == "cuda":
    # === OPTIMIZACIONES CUDA (Windows/Linux) ===
    # TF32 para GPUs Ampere+ (30xx, 40xx) - ~3x más rápido
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Benchmark para encontrar algoritmos óptimos
    torch.backends.cudnn.benchmark = True
    # Desactivar depuración para máxima velocidad
    torch.backends.cudnn.deterministic = False
    # Mostrar GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
elif DEVICE == "mps":
    # === OPTIMIZACIONES MPS (Apple Silicon) ===
    print(f"🍎 Apple Silicon (MPS)")

# Liberar memoria GPU al inicio (si está habilitado)
if USE_GPU_EMPTY_CACHE:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

# --- Global Model Initialization ---
MODEL = None



LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


def open_output_folder():
    """Opens the output folder in the system's file explorer."""
    import subprocess
    import platform
    
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", OUTPUT_DIR], check=True)
        elif platform.system() == "Windows":
            subprocess.run(["explorer", OUTPUT_DIR], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", OUTPUT_DIR], check=True)
        return f"✅ Carpeta abierta: {OUTPUT_DIR}"
    except Exception as e:
        return f"❌ Error al abrir carpeta: {e}"


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model with optimizations."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            
            # === OPTIMIZACIÓN 1: torch.compile() ===
            # Compilar el modelo para mejor rendimiento (PyTorch 2.0+)
            try:
                print("🔥 Compilando modelo con torch.compile()...")
                # Compilar solo en CUDA (MPS no soporta compile aún)
                if DEVICE == "cuda":
                    MODEL = torch.compile(MODEL, mode="reduce-overhead")
                    print("✅ Modelo compilado exitosamente")
                else:
                    print("⚠️  torch.compile() no disponible en MPS, usando modelo sin compilar")
            except Exception as e:
                print(f"⚠️  No se pudo compilar el modelo: {e}")
            
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


# === OPTIMIZACIÓN 2: Caché de embeddings ===
EMBEDDING_CACHE = {}

def get_audio_embedding(audio_path: str, exaggeration: float, model):
    """Obtiene el embedding de audio con caché para evitar recomputación."""
    cache_key = (audio_path, exaggeration)
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    print(f"🎙️ Computando embedding para: {audio_path.split('/')[-1]} (exaggeration: {exaggeration})")
    try:
        # Esto prepara los condicionales dentro del modelo
        model.prepare_conditionals(audio_path, exaggeration=exaggeration)
        # Guardamos una copia del objeto conds (que contiene los tensores del embedding)
        EMBEDDING_CACHE[cache_key] = model.conds
        return model.conds
    except Exception as e:
        print(f"⚠️ Error al preparar embedding: {e}")
        return None


# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Error: {e}")


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks at sentence boundaries for natural speech flow.
    """
    import re
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # Handle long sentences
            if len(sentence) > max_chunk_size:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                        current_chunk = f"{current_chunk} {word}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


@torch.inference_mode()  # Más eficiente que no_grad() para inferencia
def generate_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfg_weight_input: float = 0.5,
    repetition_penalty_input: float = 2.0,
    min_p_input: float = 0.05,
    progress=gr.Progress()
) -> str:
    """
    Generate audio for the given text using the TTS model.
    Supports long texts by processing them in chunks (up to 10,000 characters).
    """

    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    # Validate and truncate text
    text_input = text_input.strip()
    if not text_input:
        raise ValueError("Text input is empty.")
    
    text_input = text_input[:MAX_CHARS]

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    # Resolve audio prompt and embedding
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)
    
    # === USO DE CACHÉ DE EMBEDDINGS ===
    if chosen_prompt:
        embedding = get_audio_embedding(chosen_prompt, exaggeration_input, current_model)
        if embedding:
            current_model.conds = embedding
    
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfg_weight_input,
        "repetition_penalty": repetition_penalty_input,
        "min_p": min_p_input,
    }
    # NO pasamos audio_prompt_path porque ya seteamos current_model.conds manualmente vía caché
    # Si lo pasamos, el modelo volvería a llamar a prepare_conditionals internamente
    # No obstante, si el embedding falló, dejamos que el modelo lo intente cargar normalmente
    if not chosen_prompt:
         # Si no hay prompt, el modelo podría fallar si no tiene conds
         pass

    # Split text into chunks
    chunks = split_text_into_chunks(text_input)
    total_chunks = len(chunks)
    
    # === RESUMEN AL INICIO ===
    print(f"\n{'='*70}")
    print(f"📝 RESUMEN: {len(text_input):,} caracteres → {total_chunks} chunks (máx. {CHUNK_SIZE} chars/chunk)")
    print(f"🌐 Idioma: {language_id}")
    if chosen_prompt:
        print(f"🎤 Audio de referencia: {chosen_prompt.split('/')[-1]}")
    print(f"{'='*70}\n")

    all_wavs = []
    
    # Progreso inicial
    progress(0, desc=f"📝 Preparando {total_chunks} chunks...")
    
    import time
    start_time = time.time()
    chunk_times = []
    
    # Bucle de generación
    for chunk_idx in range(total_chunks):
        chunk_start = time.time()
        chunk_text = chunks[chunk_idx]
        
        # Vista previa del chunk actual
        preview = chunk_text[:45] if len(chunk_text) > 45 else chunk_text
        preview = preview.replace('\n', ' ')
        
        # Progreso en terminal
        if chunk_idx > 0:
            avg_time = sum(chunk_times) / len(chunk_times)
            eta = avg_time * (total_chunks - chunk_idx)
            eta_minutes = int(eta // 60)
            eta_seconds = int(eta % 60)
            print(f"📦 [{chunk_idx + 1}/{total_chunks}] '{preview}...' (ETA: {eta_minutes}:{eta_seconds:02d})")
        else:
            print(f"📦 [{chunk_idx + 1}/{total_chunks}] '{preview}...'")
        
        # Progreso en Gradio UI
        progress_pct = (chunk_idx + 1) / total_chunks
        progress(progress_pct, desc=f"🎙️ Chunk {chunk_idx + 1}/{total_chunks}: '{preview[:30]}...'")
        
        wav = current_model.generate(
            chunk_text,
            language_id=language_id,
            **generate_kwargs
        )
        # Normalizar dimensiones del tensor
        wav = wav.squeeze()
        if wav.dim() == 0:
            continue  # Ignorar tensores vacíos
        if wav.dim() == 2:
            wav = wav[0]  # Tomar solo el primer canal si es estéreo
        
        # Guardar como float32 (más rápido, convertir a int16 solo al final)
        all_wavs.append(wav.cpu())
        del wav
        
        chunk_times.append(time.time() - chunk_start)

    
    # Concatenate all chunks
    progress(1.0, desc="✅ Concatenando audio...")
    
    if len(all_wavs) == 0:
        raise ValueError("No audio chunks were generated")
    
    # Debug: mostrar formas de chunks
    print(f"\n📊 Debug - Chunks generados: {len(all_wavs)}")
    total_samples = sum(w.shape[-1] for w in all_wavs)
    print(f"   Total samples: {total_samples:,}")
    
    # Concatenar todos los chunks
    final_wav = torch.cat(all_wavs, dim=-1)
    del all_wavs
    
    # === DURACIÓN FINAL ===
    total_time = time.time() - start_time
    duration = final_wav.shape[-1] / current_model.sr
    
    print(f"\n{'='*70}")
    print(f"✅ GENERACIÓN COMPLETA")
    print(f"   📊 Chunks procesados: {total_chunks}")
    print(f"   🎵 Total samples: {final_wav.shape[-1]:,}")
    print(f"   ⏱️  Duración del audio: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   ⚡ Tiempo de generación: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   📈 Velocidad: {duration/total_time:.2f}x realtime")
    print(f"{'='*70}\n")
    
    progress(1.0, desc=f"✅ ¡Completado! Duración: {duration:.1f}s | Velocidad: {duration/total_time:.1f}x")
    
    # Guardar audio a archivo WAV temporal
    import tempfile
    import scipy.io.wavfile as wavfile
    
    # Convertir a int16 solo al final
    audio_numpy = final_wav.numpy()
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    del final_wav, audio_numpy
    
    # Guardar audio en carpeta persistente con timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"audio_{timestamp}.wav")
    wavfile.write(output_path, current_model.sr, audio_int16)
    
    print(f"📤 Audio guardado: {output_path}")
    print(f"   Tamaño: {len(audio_int16) * 2 / 1e6:.1f} MB")
    
    # Liberar memoria
    del audio_int16
    
    # === LIMPIEZA SELECTIVA DE CACHÉ ===
    if AUTO_CLEAN_CACHE:
        print(f"\n🧹 Limpiando caché temporal...")
        import subprocess
        import glob
        
        try:
            # 1. Limpiar archivos temporales de Gradio (excepto el audio generado)
            # Usamos glob para encontrar directorios temporales de Gradio
            gradio_temp_dirs = glob.glob("/private/var/folders/*/T/gradio/*")
            for temp_dir in gradio_temp_dirs:
                if os.path.isdir(temp_dir) and temp_file.name not in temp_dir:
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
            
            # 2. NO BORRAR modelos de Huggingface - solo limpiar lockfiles y temp
            hf_cache = os.path.expanduser('~/.cache/huggingface')
            if os.path.exists(hf_cache):
                # Usamos find via subprocess para eficiencia en árboles grandes
                # NO usar capture_output junto con stdout/stderr para evitar conflictos
                subprocess.run(['find', hf_cache, '-name', '*.lock', '-delete'], 
                              check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['find', hf_cache, '-type', 'f', '-name', 'tmp*', '-delete'], 
                              check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 3. Limpiar solo tarballs de conda (mantener paquetes instalados)
            subprocess.Popen(['conda', 'clean', '--tarballs', '-y'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 4. Limpiar pip cache (en background)
            subprocess.Popen(['pip', 'cache', 'purge'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"✅ Caché temporal limpiada (modelos preservados)")
        except Exception as e:
            print(f"⚠️ Error limpiando caché: {e}")

    
    
    return output_path






# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox Studio
        **Professional Multilingual Text-to-Speech Engine**
        
        Generate high-quality multilingual speech from text with reference audio styling.
        Supports up to **10,000 characters** with automatic chunk processing.
        """
    )
    
    gr.Markdown(get_supported_languages_display())
    
    with gr.Row():
        with gr.Column(scale=1):
            initial_lang = "es"
            
            language_id = gr.Dropdown(
                choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                value=initial_lang,
                label="🌐 Language",
                info="Select the language for synthesis"
            )
            
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎤 Reference Audio (Optional)",
                value=default_audio_for_ui(initial_lang)
            )
            
            gr.Markdown(
                "💡 **Tip**: Match reference audio language with selected language for best results.",
                elem_classes=["audio-note"]
            )
            
            exaggeration = gr.Slider(
                0.25, 2, step=0.05, 
                label="🎭 Exaggeration", 
                value=0.5,
                info="Neutral = 0.5"
            )
            
            cfg_weight = gr.Slider(
                0.2, 1, step=0.05, 
                label="⚡ CFG/Pace", 
                value=0.5
            )

            with gr.Accordion("⚙️ Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random Seed (0 = random)")
                temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
                repetition_penalty = gr.Slider(1.0, 10.0, step=0.1, label="Repetition Penalty", value=2.0)
                min_p = gr.Slider(0.01, 0.5, step=0.01, label="Min P", value=0.05)

        with gr.Column(scale=2):
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label=f"📝 Text to Synthesize (max {MAX_CHARS:,} characters)",
                lines=10,
                max_lines=20
            )
            
            run_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
            
            open_folder_btn = gr.Button("� Open Output Folder", size="sm")
            
            audio_output = gr.Audio(label="� Generated Audio")
            
            open_folder_btn.click(
                fn=open_output_folder,
                inputs=[],
                outputs=[]
            )

    def on_language_change(lang, current_ref, current_text):
        return default_audio_for_ui(lang), default_text_for_ui(lang)

    language_id.change(
        fn=on_language_change,
        inputs=[language_id, ref_wav, text],
        outputs=[ref_wav, text],
        show_progress=False
    )

    run_btn.click(
        fn=generate_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            repetition_penalty,
            min_p,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
