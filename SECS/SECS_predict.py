from SECS.resemblyzer.audio import preprocess_wav
from SECS.resemblyzer.voice_encoder import VoiceEncoder
from pathlib import Path
import numpy as np
import os
import csv

def predictSECS(dir_path:str,refer_path:str,output_dir:str):
    # Initialize the encoder
    encoder = VoiceEncoder()

    # Load and preprocess the reference audio
    reference_path = Path(refer_path)
    reference_wav = preprocess_wav(reference_path)

    # Embed the reference audio
    reference_embed = encoder.embed_utterance(reference_wav)

    # Load and preprocess the inference audios
    inference_paths = list(Path(dir_path).glob("*.wav"))
    inference_wavs = [preprocess_wav(wav_path) for wav_path in inference_paths]

    # Embed the inference audios
    inference_embeds = np.array([encoder.embed_utterance(wav) for wav in inference_wavs])

    # Compute the similarity between the reference and each inference audio
    similarities = np.inner(reference_embed, inference_embeds)

    folder_name = os.path.basename(os.path.normpath(dir_path))
    csv_file_path = os.path.join(output_dir, f"{folder_name}_SECS.csv")

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Inference File', 'Similarity'])
        for path, similarity in zip(inference_paths, similarities):
            print(f"Similarity between {reference_path.name} and {path.name}: {similarity}")
            writer.writerow([path.name, similarity])

    print(f"Results saved to {csv_file_path}")