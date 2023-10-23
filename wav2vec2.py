import os
import pandas as pd
import torchaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import faiss
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_vectors(vectors):
    # Normalize vectors to unit length
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def identify_speakers_and_generate_csv(directory_path, output_csv_path, similarity_threshold=0.6):
    try:
        logging.info("Loading SpeakerRecognition model...")
        # Load the pre-trained SpeakerRecognition model
        model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_emozr")
        logging.info("SpeakerRecognition model loaded successfully.")

        # Initialize an empty list to store (file_name, speaker_id) pairs
        file_speaker_pairs = []
        speaker_id = 0

        logging.info("Processing MP3 files and extracting embeddings...")
        # Process MP3 files and extract embeddings
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp3"):
                logging.info(f"Processing file: {filename}")
                # Obtain speaker embedding (xvector)
                waveform, sample_rate = torchaudio.load(os.path.join(directory_path, filename))
                embedding = model.encode_batch(waveform)
                
                # Convert the PyTorch tensor to NumPy array and flatten to 1D
                embedding_np = embedding.numpy().reshape(-1)

                # Normalize the embedding vector
                embedding_np_normalized = normalize_vectors(embedding_np.reshape(1, -1))[0]

                # Calculate cosine similarity with existing speakers using Faiss
                matched_speaker_id = None

                if len(file_speaker_pairs) != 0:
                    existing_embeddings_np = np.array([existing_embedding for _, _, existing_embedding, _ in file_speaker_pairs], dtype=np.float32)

                    # Normalize existing embeddings
                    existing_embeddings_np_normalized = normalize_vectors(existing_embeddings_np)

                    query_embedding_np = np.expand_dims(embedding_np_normalized, axis=0).astype(np.float32)

                    # Setup Faiss index
                    dimension = existing_embeddings_np.shape[1]  # Dimension of the feature vector
                    index = faiss.IndexFlatL2(dimension)  # L2 distance is used for cosine similarity
                    index.add(existing_embeddings_np_normalized)

                    # Perform similarity search using Faiss
                    _, similarity_indices = index.search(query_embedding_np, 1)

                    similarity_score = similarity_indices[0][0]
                    existing_speaker_id = file_speaker_pairs[similarity_indices[0][0]][1]

                    logging.info(f"Speaker ID {existing_speaker_id} - Similarity Score: {similarity_score}")

                    if similarity_score >= similarity_threshold:
                        matched_speaker_id = existing_speaker_id

                logging.info(f"Length of File Speaker Pairs {len(file_speaker_pairs)}")

                # Assign a new speaker ID if no match is found
                if matched_speaker_id is None:
                    new_speaker_id = speaker_id + 1
                    file_speaker_pairs.append((filename, new_speaker_id, embedding_np_normalized, new_speaker_id))
                    logging.info(f"New speaker ID assigned: {new_speaker_id}")
                    speaker_id += 1
                else:
                    file_speaker_pairs.append((filename, matched_speaker_id, embedding_np_normalized, matched_speaker_id))
                    logging.info(f"Matched with existing speaker ID: {matched_speaker_id}")

        logging.info("Creating CSV file...")
        # Create a DataFrame from the list of (file_name, speaker_id) pairs
        df = pd.DataFrame(file_speaker_pairs, columns=['File Name', 'Speaker ID'])

        # Write the DataFrame to the CSV file using pandas
        df.to_csv(output_csv_path, index=False)

        logging.info(f"CSV file '{output_csv_path}' generated successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# Example usage
directory_path = "cv-corpus-5.1-2020-06-22/rw/selected_clips"
output_csv_path = "output.csv"
identify_speakers_and_generate_csv(directory_path, output_csv_path)
