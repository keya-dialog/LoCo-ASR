# Regularization enhancements

1. Show gain from joint decoding - DONE
2. Scaling experiments (wider, deeper, wider + more tokens) + regularization's - WIP
3. freeze everything add there another scalar, layer vocab size vs full matrix - WIP

# Regularization real scenario

1. Prepare english corpuses - 5500h (apply specaug+speed perturbation)
   1. Librispeech - 1000h - DONE
   2. Tedlium - 450h - DONE
   3. Voxpopuli - 550h - DONE
   4. Fisher+SW - 2000h - DONE
   5. Common voice - 1500h - DONE
2. Prepare evaluation datasets
   1. Common voice - DONE
   2. FLEURS - DONE
   3. Librispeech test clean - DONE
   4. Librispeech test other - DONE
   5. SWITCHBOARD - DONE
   6. Tedlium - DONE
   7. Voxpopuli - DONE
   8. WSJ - DONE
3. Prepare common normalizations and evaluation protocol - DONE
4. Train small (12,6,256h) and big (16,8,512h) -speed perturb (0.9,1.1) + specaug 100 epochs
5. Deeper check - layer 6
6. Logit averaging on validation

# Multichannel ASR

1. consult eperiments with Lada
2. prepare chime dataset

# Ebranchformer extension

1. Try attention mixing

# cgMLP only encoder

1. try how it works on ted

# EU ASR

1. Implement CTC trainer - DONE
2. Prepare CZ datasets - DONE

   ```
   COMMON_VOICE_TRAIN=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_CommonVoice_v13/data/train_true-case-punct
   COMMON_VOICE_DEV=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_CommonVoice_v13/data/dev_true-case-punct
   COMMON_VOICE_TEST=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_CommonVoice_v13/data/test_true-case-punct
   
   LDC_BROADCAST_TRAIN=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_BroadcastNewsConvs_LDC/data/Czech-BNC-LDC_train
   LDC_BROADCAST_DEV=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_BroadcastNewsConvs_LDC/data/Czech-BNC-LDC_dev
   LDC_BROADCAST_TEST=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_BroadcastNewsConvs_LDC/data/Czech-BNC-LDC_test
   
   PARCZECH_TRAIN_300=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_ParCzech_3.0/kaldi_data/parczech30_train_true-case-punct_300h
   PARCZECH_DEV=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_ParCzech_3.0/kaldi_data/parczech30_dev_true-case-punct
   PARCZECH_TEST=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_ParCzech_3.0/kaldi_data/parczech30_test_true-case-punct
   
   PDTSC_TRAIN=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_PDTSC_2.0/data/Czech-PDTSC20_train
   PDTSC_DEV=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_PDTSC_2.0/data/Czech-PDTSC20_dev
   PDTSC_TEST=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_PDTSC_2.0/data/Czech-PDTSC20_test
   ```

3. Finetune checkpoints (Wav2vec2 - transformer, conformer, ebranchformer, zipformer) with CTC head - WIP
4. Prepare augmentations (RIRs, speed perturbation, specAugment, noise)

   ```
   http://openslr.org/28/
   noises /mnt/matylda6/szoke/CHIME/CHiME-7/dataaugmentation/lists/all_NOISE.list
   rirs /mnt/matylda6/szoke/CHIME/CHiME-7/dataaugmentation/lists/all_RIR.list
   ```

5. Reproduce one of Junyi's recipes with Zipformer,
   Ebranchformer (https://github.com/s3prl/s3prl/blob/main/example/run_asr.sh)
6. Prepare casual pretraining

### Further notes

Santosh:

1. Multiobjective pretraining with simultaneous ASR finetunning - representations matching

Lukas:

1. Reproduce wavlm with conformer architecture
2. Monitor more statictics like tokens coverage

# Whisper CTC

1. Prepare CTC training with Whisper - Done
2. Implement joint decoding with Whisper 

# Recurrent feature extraction

1. Discuss with Hynek