# Regularization enhancements

1. Show gain from joint decoding
2. Scaling experiments (wider, deeper, wider+more tokens) + regularization's - WIP
3. freeze everything add there another layer - regularization

# Regularization real scenario

1. Prepare english corpuses - 5500h (apply specaug+speed perturbation)
   1. Librispeech - 1000h - READY
   2. Tedlium - 450h - READY
   3. Voxpopuli - 550h - READY
   4. Fisher+SW - 2000h - WIP
   5. Common voice - 1500h - READY
2. Prepare evaluation datasets
   1. Common voice - READY
   2. FLEURS - WIP
   3. Librispeech test clean - READY
   4. Librispeech test other - READY
   5. SWITCHBOARD - WIP
   6. Tedlium - READY
   7. Voxpopuli - READY
   8. WSJ
3. Prepare common normalizations and evaluation protocol
4. Train medium model - 16l 512 8l 512

# Multichannel ASR

1. consult eperiments with Lada
2. prepare chime dataset

# Ebranchformer extension

1. Try attention mixing

# cgMLP only encoder - try how it works

# EU ASR

1. wavlm
2. pretraining should be optimal
3. only SSL afterwards supervised
4. try casual pretraining
5. prepare more statictics like tokens coverage
6. reproduce one of Junyis recipes with Zipformer (https://github.com/s3prl/s3prl/blob/main/example/run_asr.sh)
7. CTC training on czech corpora (prepare KarelV datasets)
8. Apply augmentations (RIRs, speed perturbation, specAugment, noise)
