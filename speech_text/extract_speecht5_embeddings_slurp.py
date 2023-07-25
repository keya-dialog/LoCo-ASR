import os
import pickle
import argparse

from slurp_data import SLURPDataset
from intent_classes import ALL_CLASSES

import torch
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5ForTextToSpeech, set_seed

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

parser = argparse.ArgumentParser(description='Extract embeddings from SLURP data with SpeechT5')
parser.add_argument('--modality', '-m', choices=['text', 'audio'], default='text', required=True, help='Modality (text or audio).')
parser.add_argument('--split', '-s', choices=['train', 'devel', 'test', 'train_synthetic'], default='train', required=True, help='Split (train or test).')
args = parser.parse_args()
modality = args.modality
split = args.split

print(f"Extracting {modality} embeddings from SLURP {split} set using SpeechT5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

data_path = "slurp"

print("STARTED!!!")
slurp_dataset = SLURPDataset(data_path, mode=split, task="intent")
print("DATA IS NOT CAUSING OUT OF MEMORY!!!!!!!!!!!!!!")

print(f"{split} set size: {len(slurp_dataset)}")

CLASSES = ALL_CLASSES #slurp_dataset.intents
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(CLASSES)
label_binarizer = LabelBinarizer()
onehot_labels = label_binarizer.fit_transform(numerical_labels)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")

def collate_fn(batch):
    slurp_ids, texts, audios, sample_rates, tasks = zip(*batch)
    
    print(type(audios))
    print(type(audios[0]))
    print(audios[0].shape)
    print(audios.shape)
    print(aaddd)
    input_texts = processor(text=texts, return_tensors='pt', padding="longest").to(device)
    input_audios = processor(audio=audios, sampling_rate=sample_rates[0], return_tensors="pt", padding="longest").to(device)

    targets = label_encoder.transform(tasks)
    targets = label_binarizer.transform(targets)
 
    return slurp_ids, input_texts, input_audios, sample_rates, targets

batch_size = 2 #16
data_loader = DataLoader(slurp_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

embeddings_folder = os.path.join("extracted", "speecht5")

if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

save_folder = os.path.join(os.path.join(embeddings_folder, split), modality)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if modality == "text":
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    print("Loaded model")
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            slurp_ids, texts, _, _, targets = data
            n = texts['input_ids'].shape[0]
            speaker_embeddings = torch.zeros((n, 512)).to(device)
            set_seed(555)
            decoder_input_values = torch.zeros((n, 1024, 80)).to(device)
            out = model(**texts, speaker_embeddings=speaker_embeddings, decoder_input_values=decoder_input_values)
            embeddings = out.encoder_last_hidden_state.cpu().detach().numpy()


            #save_embeddings = {}
            for slurp_id, text_embedding, target in zip(slurp_ids, embeddings, targets):
                with open(os.path.join(save_folder, f'{slurp_id}_embedding_and_target.pickle'), 'wb') as handle:
                    pickle.dump({"id":slurp_id, "embedding":text_embedding, "target":target}, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #print("ID:", slurp_id)
                #print("Embedding:", text_embedding.shape)
                #print("Target:", target)

else:
    # SPEECH
    model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr").to(device)
    model.eval()
    print("Loaded model")
    with torch.no_grad():
        for data in data_loader:
            slurp_ids, _, audios, sample_rates, targets = data
            
            predicted_ids_speech = model.generate(**audios, max_length=450)
            out = model(**audios, decoder_input_ids=predicted_ids_speech)
            embeddings = out.encoder_last_hidden_state.cpu().detach().numpy()
            
            for slurp_id, speech_embedding, target in zip(slurp_ids, embeddings, targets):
                with open(os.path.join(save_folder, f'{slurp_id}_embedding_and_target.pickle'), 'wb') as handle:
                    pickle.dump({"id":slurp_id, "embedding":speech_embedding, "target":target}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #print("ID:", slurp_id)
                #print("Embedding:", speech_embedding.shape)
                #print("Target:", target)
        
print("Done!")