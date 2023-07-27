import os
import jsonlines
from torch.utils.data import Dataset
import librosa

class SLURPDataset(Dataset):
    def __init__(self, data_path, mode="train", task="intent"):
        """
        mode can be: train, test, devel, train_synthetic
        task can be: intent, action, enteties, scenario, sentence_annotation, tokens 
        """
        self.data_path = data_path
        self.mode = mode
        self.task = task
        self.intents = []
        #self.intents_before = []
        self.dataset = self.prepare_data()

    def prepare_data(self):
        text_path = os.path.join(self.data_path, "dataset/slurp")
        audio_path = os.path.join(self.data_path, "audio")
        
        #Read text
        text_file = self.mode + ".jsonl"
        text_file_path = os.path.join(text_path, text_file)
        
        audio_mode = "slurp_real"
        if self.mode == "train_synthetic":
            audio_mode = "slurp_synth"
        
        audio_files_path = os.path.join(audio_path, audio_mode)
        audio_files_list = os.listdir(audio_files_path)
        
        slurp_dataset = {}
        intents = []
        with jsonlines.open(text_file_path) as file:
            for i,item in enumerate(file):
                #!!!!! For now only retrieve one recording file from the sample LOOK FOR HEADSET!!!!!
                recording_file = next((audio_name['file'] for audio_name in item["recordings"] if "headset" in audio_name), item["recordings"][0]['file'])
                #recording_file = item["recordings"][0]['file']

                #if recording_file in audio_files_list:
                audio_example = os.path.join(audio_files_path, recording_file)
                #audio_data, sample_rate = librosa.load(audio_example, sr=16000) #Originally 22050 Hz but now 16k

                slurp_dataset[i] = {"slurp_id":item['slurp_id'], "text":item, "audio":{"path":audio_example, "sample_rate":16000}}
                intents.append(item["intent"])

        if self.task == "intent":
            self.intents = list(set(intents))
            #self.intents_before = intents

        return slurp_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        slurp_id = self.dataset[idx]['slurp_id']
        text = self.dataset[idx]['text']['sentence']
        audio = self.dataset[idx]['audio']['path']
        sampling_rate = self.dataset[idx]['audio']['sample_rate']
        
        downstream_task = self.dataset[idx]['text'][self.task]
        
        return slurp_id, text, audio, sampling_rate, downstream_task
