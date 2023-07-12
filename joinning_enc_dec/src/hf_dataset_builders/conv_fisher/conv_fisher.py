import logging
import math
import os
from itertools import groupby
from typing import Iterable, Tuple, Union, List
import datasets
import kaldiio
import numpy as np
import librosa

_DATA_PREFIX = "data_topic_split"
_FILEPATHS = {
    "train": {
        "feats": f"{_DATA_PREFIX}/train/wav.scp",
        "segments": f"{_DATA_PREFIX}/train/segments",
        "transcripts": f"{_DATA_PREFIX}/train/text",
        "channels2recordings": f"{_DATA_PREFIX}/train/reco2file_and_channel"
    },
    "train_500": {
        "feats": f"{_DATA_PREFIX}/train_500/wav.scp",
        "segments": f"{_DATA_PREFIX}/train_500/segments",
        "transcripts": f"{_DATA_PREFIX}/train_500/text",
        "channels2recordings": f"{_DATA_PREFIX}/train_500/reco2file_and_channel"
    },
    "validation": {
        "feats": f"{_DATA_PREFIX}/dev/wav.scp",
        "segments": f"{_DATA_PREFIX}/dev/segments",
        "transcripts": f"{_DATA_PREFIX}/dev/text",
        "channels2recordings": f"{_DATA_PREFIX}/dev/reco2file_and_channel"
    },
    "dev_6": {
        "feats": f"{_DATA_PREFIX}/dev_6/wav.scp",
        "segments": f"{_DATA_PREFIX}/dev_6/segments",
        "transcripts": f"{_DATA_PREFIX}/dev_6/text",
        "channels2recordings": f"{_DATA_PREFIX}/dev_6/reco2file_and_channel"
    },
    "test": {
        "feats": f"{_DATA_PREFIX}/test/wav.scp",
        "segments": f"{_DATA_PREFIX}/test/segments",
        "transcripts": f"{_DATA_PREFIX}/test/text",
        "channels2recordings": f"{_DATA_PREFIX}/test/reco2file_and_channel"
    },
}

_TOPIC_LABELS = "etc/topic_labels.txt"
_RECORDING_IDS = "etc/rec_ids.txt"


class Fisher(datasets.GeneratorBasedBuilder):
    """Dataset builder for Fisher dataset"""
    DEFAULT_WRITER_BATCH_SIZE = 50  # the default size of the batch may not fit in memory

    def __init__(self, metadata_dir: os.PathLike, sampling_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.data_dir = metadata_dir

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            supervised_keys=None,
            homepage="",
        )

    def _prepare_split_single(
            self,
            gen_kwargs: dict,
            fpath: str,
            file_format: str,
            max_shard_size: int,
            split_info: datasets.SplitInfo,
            check_duplicate_keys: bool,
            job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        self.info.features = None  # Disable unnecessary type check and conversion that slows generation
        return super()._prepare_split_single(gen_kwargs, fpath, file_format, max_shard_size, split_info,
                                             check_duplicate_keys,
                                             job_id)

    def _split_generators(self, _):
        """Generate dataset splits"""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=self._fetch_split_meta("train"),
            ),
            datasets.SplitGenerator(
                name="train_500",
                gen_kwargs=self._fetch_split_meta("train_500"),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=self._fetch_split_meta("validation"),
            ),
            datasets.SplitGenerator(
                name="dev_6",
                gen_kwargs=self._fetch_split_meta("dev_6"),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=self._fetch_split_meta("test"),
            ),
        ]

    def _fetch_split_meta(self, split: str):
        """Fetch split meta data from kaldi-like dataset"""
        with open(os.path.join(self.data_dir, _FILEPATHS[split]["transcripts"])) as f:
            texts = dict(map(lambda line: line.strip().split(maxsplit=1), f))  # creates (segment_id -> text) mapping

        with open(os.path.join(self.data_dir, _FILEPATHS[split]["segments"])) as f:
            segments = dict(map(lambda s: self._parse_segment_info(*s.strip().split()),
                                f))  # creates (segment_id -> wav_id, start, end) mapping

        with open(os.path.join(self.data_dir, _TOPIC_LABELS)) as f:
            topics = f.read().splitlines()
        with open(os.path.join(self.data_dir, _RECORDING_IDS)) as f:
            ids = f.read().splitlines()
        topic_mapping = dict(zip(ids, topics))

        with open(os.path.join(self.data_dir, _FILEPATHS[split]["channels2recordings"])) as f:
            recording_mappings = dict(
                map(lambda line: line.strip().split()[:2], f))  # creates (recording_channel -> recording) mapping

        # load kaldiio feature generator
        featfile = os.path.join(self.data_dir, _FILEPATHS[split]["feats"])
        feats_generator = kaldiio.load_scp(featfile)
        segments = [(*segments[uttid], uttid, transcript) for (uttid, transcript) in texts.items()]
        grouped_by_recordings = [(k, list(v)) for k, v in groupby(sorted(segments), key=lambda segment: segment[0])]
        return {"recordings": grouped_by_recordings, "features": feats_generator, "topic_mapping": topic_mapping,
                "recording_mappings": recording_mappings}

    def _extract_segments(self, sorted_segments, audio):
        n_segments = len(sorted_segments)
        output = {"n_turns": n_segments, "audio": [], "text": [], "utt_id": [], "audio_len": []}
        for _, start, end, uttid, transcript in sorted_segments:
            output["text"].append(self.preprocess_text(transcript))
            output["audio"].append(self._crop_audio(audio, self.sampling_rate, start, end))
            output["utt_id"].append(uttid)
            output["audio_len"].append(end - start)
        return output

    def _generate_examples(self, recordings, features, topic_mapping, recording_mappings):
        """Generator for split examples fetching"""
        for recording, segments in recordings:
            sr, audio = features[recording]
            topic = topic_mapping[recording_mappings[recording]]
            if audio.dtype == np.int16:
                audio = librosa.util.buf_to_float(audio, n_bytes=audio.dtype.itemsize)
            else:
                raise ValueError('Data type of input audio is not int16.')
            if len(audio.shape) > 1:
                raise ValueError(f'Recording {recording} does not have single channel.')
            elif sr != self.sampling_rate or len(audio.shape) > 1:
                logging.debug(f'Resampled {recording} from {sr} to {self.sampling_rate}')
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            sorted_segments = sorted(segments, key=lambda x: x[1])
            segments = self._extract_segments(sorted_segments, audio)
            yield recording, {"recording": recording, "topic": topic, **segments,
                              "n_turns": len(sorted_segments)}

    @staticmethod
    def _parse_segment_info(segment_key, uri, start, end):
        return segment_key, (uri, float(start), float(end))

    @staticmethod
    def _crop_audio(audio, sampling_rate, start, end):
        return audio[math.floor(sampling_rate * start): math.ceil(end * sampling_rate)]

    @staticmethod
    def preprocess_text(utterance_batch: List[str]):
        return utterance_batch
