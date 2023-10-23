import os
import pandas as pd
import torchaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
import logging
from multiprocessing import Pool

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(filename, directory_path, model, similarity_threshold):
    try:
        logging.info(f"Processing file: {filename}")
        waveform, sample_rate = torchaudio.load(os.path.join(directory_path, filename))
        embedding = model.encode_batch(waveform)
        embedding_np = embedding.numpy().reshape(-1)
        return filename, embedding_np
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None, None

def identify_speakers_and_generate_csv(directory_path, output_csv_path, similarity_threshold=0.6):
    try:
        logging.info("Loading SpeakerRecognition model...")
        model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_emozr")
        logging.info("SpeakerRecognition model loaded successfully.")

        file_speaker_pairs = []

        logging.info("Processing WAV files and extracting embeddings...")

        # Use multiprocessing to process files concurrently
        with Pool() as pool:
            results = [result for result in pool.starmap(process_file, [(filename, directory_path, model, similarity_threshold) for filename in os.listdir(directory_path) if filename.endswith(".mp3")])]

        for filename, embedding_np in results:
            if embedding_np is not None:
                matched_speaker_id = None
                for _, _, existing_embedding, existing_speaker_id in file_speaker_pairs:
                    similarity_score = cosine_similarity([embedding_np], [existing_embedding])[0][0]
                    if similarity_score >= similarity_threshold:
                        matched_speaker_id = existing_speaker_id
                        break

                if matched_speaker_id is None:
                    new_speaker_id = len(file_speaker_pairs) + 1
                    file_speaker_pairs.append((filename, new_speaker_id, embedding_np, new_speaker_id))
                    logging.info(f"New speaker ID assigned: {new_speaker_id}")
                else:
                    file_speaker_pairs.append((filename, matched_speaker_id, embedding_np, matched_speaker_id))
                    logging.info(f"Matched with existing speaker ID: {matched_speaker_id}")

        logging.info("Creating CSV file...")

        df = pd.DataFrame(file_speaker_pairs, columns=['File Name', 'Speaker ID'])
        df.to_csv(output_csv_path, index=False)

        logging.info(f"CSV file '{output_csv_path}' generated successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# Example usage
directory_path = "cv-corpus-5.1-2020-06-22/rw/selected_clips"
output_csv_path = "output.csv"
identify_speakers_and_generate_csv(directory_path, output_csv_path)
