from transformers import pipeline
import datasets
from safe_gpu import safe_gpu
from transformers.pipelines.pt_utils import KeyDataset
from jiwer import wer

safe_gpu.claim_gpus(1)
import torch
import pickle
from tqdm import tqdm
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    chunk_length_s=20,
    device=device,
    batch_size=16
)

dataset = datasets.load_from_disk('/mnt/matylda5/xpolok03/projects/LoCo-ASR/datasets/fisher',
                                  keep_in_memory=False).with_format("np")
generated = []
for out in tqdm(pipe(KeyDataset(dataset['dev_6'], 'input_values'))):
    generated.append(out["text"].lower().strip())
with open('dev_6', 'wb') as f:
    pickle.dump(generated, f)
dev_wer = wer(list(dataset['dev_6']['labels']), generated)
print(dev_wer)

generated = []
for out in tqdm(pipe(KeyDataset(dataset['test'], 'input_values'))):
    generated.append(out["text"].lower().strip())
with open('test', 'wb') as f:
    pickle.dump(generated, f)
test_wer = wer(list(dataset['test']['labels']), generated)
print(test_wer)
