# SpeechText Intent Classification with SpeechT5

This repository contains code and instructions for training a classifier for Intent Classification on the SLURP dataset using a Speech-Text Joint Representation Model called SpeechT5. The model leverages the power of HuggingFace's pre-trained T5 model, fine-tuned on speech and text data, to create joint embeddings that capture both audio and text information for improved intent classification.

## Folder Structure

The folder structure for this repository is organized as follows:
speech_text/
├── checkpoints (generated)/
│   ├── base/
│   │   ├── audio/
│   │   │   └── attention/
│   │   │       └── checkpoints...
│   │   └── text/
│   │       └── attention/
│   │           └── checkpoints...
│   └── fine_tuned/
│       ├── same as base
|
├── extracted (generated)/
│   └── speecht5_base/
│       ├── devel/
│       │   ├── audio/
│       │   │   └── embeddings.pickle
│       │   └── text/
│       │       └── embeddings.pickle
│       ├── test/
│       │   └── same as devel
│       ├── train/
│       │   └── same as devel
│       └── train_synthetic/
│           └── same as devel
├── notebooks/
│   └── .ipynb ...
├── results (generated)/
│   └── base/
│       ├── audio/
│       │   ├── plots/
│       │   └── logs/
│       └── text/
│           └── same as audio
├── slurp (downloaded)/
│   └── the dataset folder
├── scripts.py
└── README.md (you are here)

## Running the code
0. **Install**
   - If you have datasets and transformers packages from HuggingFace, the code should work fine. Nevertheless, do pip install -r requirements.txt

1. **Download Data**
   - First, download the SLURP dataset by following the instructions provided in the repository: [https://github.com/pswietojanski/slurp](https://github.com/pswietojanski/slurp)

2. **Using SpeechT5 Fine-tuned from HuggingFace**
   - To extract embeddings from the SLURP dataset using the SpeechT5 model fine-tuned from HuggingFace, run the script `extract_speecht5_embeddings_slurp.py`. You can specify the modality with the `-m` flag (text, audio) and the split with the `-s` flag (train, test, devel, train_synthetic).

   - Next, train a classifier for Intent Classification on the SLURP dataset using the extracted embeddings. Run the script `train_classifier.py`. You can specify the modality with the `-m` flag (text, audio), the model version with the `-v` flag (base, fine_tuned), and the pooling method with the `-p` flag (attention, max, avg).

3. **Using SpeechT5 Base Self-Supervised from Microsoft**
   - Use the Jupyter notebook `4. Save Mappings and Loading SpeechT5 Base.ipynb` to map the SpeechT5 Base from Microsoft to the HuggingFace version.

   - Then, follow a similar process as in Step 2, but use the following scripts:
     - `extract_speecht5_base_embeddings_slurp.py` to extract embeddings using SpeechT5 Base from Microsoft.
     - `train_classifier.py` to train the Intent Classification classifier using the extracted embeddings.

## Work in Progress
- [ ] Add support for the Speech version of MultiWoZ dataset.
- [ ] Include the SAMU-XLS-R model for enhanced intent classification performance.
- [ ] Add an option to fine-tune the base model for further customization and better results.

Feel free to explore the provided notebooks, scripts, and the extracted embeddings to improve the classifier's performance or adapt it for your own projects. If you encounter any issues or have suggestions, please open an issue in the repository.