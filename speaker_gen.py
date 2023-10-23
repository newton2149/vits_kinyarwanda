import csv
import logging
import pandas as pd
import torchaudio
from transformers import AutoModel
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def identify_speakers_and_generate_ids_and_update_tsv(wavs_dir, output_tsv_file, similarity_threshold=0.9):
    # Check if CUDA is available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Wav2Vec2 model to the specified device
    logging.info("Loading the Wav2Vec2 model...")
    wav2vec2_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to(device)

    # Creating a list of filenames and texts
    logging.info("Creating a list of filenames and texts...")
    filenames = []
    texts = []
    for filename in os.listdir(wavs_dir):
        filepath = os.path.join(wavs_dir, filename)
        if os.path.isfile(filepath) and filename.endswith(".wav"):
            filenames.append(filepath)
            texts.append("")

    # Extracting speaker embeddings from the audio recordings
    logging.info("Extracting speaker embeddings from the audio recordings...")
    speaker_embeddings = []
    speaker_ids = {}  # Store unique speaker IDs for grouping
    current_speaker_id = 1  # Initial speaker ID

    for filename in filenames:
        logging.info(f"Processing file: {filename}")

        # Load the audio waveform
        audio_tensor, sample_rate = torchaudio.load(filename)

        # Convert the audio tensor and the model to CUDA if available
        audio_tensor = audio_tensor.to(device)
        
        with torch.no_grad():
            speaker_embedding = wav2vec2_model(audio_tensor).last_hidden_state[:, 0]

        # Convert the speaker embedding to a NumPy array
        speaker_embedding = speaker_embedding.cpu().numpy()

        # Check for similarity with existing speakers
        matching_speaker_id = None
        for speaker_id, reference_embedding in speaker_ids.items():
            similarity_score = cosine_similarity([speaker_embedding], [reference_embedding])[0][0]
            if similarity_score >= similarity_threshold:
                matching_speaker_id = speaker_id
                break

        if matching_speaker_id is not None:
            # Assign the matching speaker's ID
            logging.info(f"Matching speaker found. Assigning speaker ID: {matching_speaker_id}")
            speaker_ids[matching_speaker_id] = np.maximum(speaker_embedding, reference_embedding)
            speaker_embeddings.append(speaker_embedding)
        else:
            # Assign a new speaker ID
            logging.info(f"No matching speaker found. Assigning new speaker ID: {current_speaker_id}")
            speaker_ids[current_speaker_id] = speaker_embedding
            speaker_embeddings.append(speaker_embedding)
            current_speaker_id += 1

    # Create a pandas DataFrame with the results
    logging.info("Creating a pandas DataFrame with the results...")
    df = pd.DataFrame({"path": filenames, "sentence": texts, "speaker_id": [speaker_id for _ in filenames]})

    # Saving the DataFrame to a tsv file
    logging.info(f"Saving the DataFrame to '{output_tsv_file}'...")
    df.to_csv(output_tsv_file, sep="\t", index=False)

    # Updating the train.tsv file
    logging.info("Updating the train.tsv file...")
    df_train = pd.read_csv("cv-corpus-5.1-2020-06-22/rw/train_updated.tsv", sep='\t')  # Assuming train.tsv exists
    for i, row in df_train.iterrows():
        filename = row['path']
        speaker_id = df.loc[df['path'] == filename, 'speaker_id'].iloc[0]
        df_train.at[i, 'speaker_id'] = speaker_id

    # Saving the updated train.tsv file
    df_train.to_csv("train_2.tsv", sep='\t', index=False)

if __name__ == "__main__":
    wavs_dir = "/cv-corpus-5.1-2020-06-22/rw/updated_wavs"
    output_tsv_file = "cv-corpus-5.1-2020-06-22/rw/speaker_ids.tsv"
    similarity_threshold = 0.9  # Adjust the similarity threshold as needed

    identify_speakers_and_generate_ids_and_update_tsv(wavs_dir, output_tsv_file, similarity_threshold)
