# from typing import Optional
#
# from torch import nn
# from transformers import AutoConfig, PretrainedConfig, \
#     SpeechEncoderDecoderConfig
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import (
#     logging,
# )
#
# from context_aware.decoder import GPT2WithContextLMHeadModel
# from context_aware.encoder import Wav2Vec2WithContextForCTC
# from per_utterance.models import JointCTCAttentionEncoderDecoder, JointCTCAttentionEncoderDecoderConfig
#
# logger = logging.get_logger("transformers")
#
#
#
#
# class JointCTCAttentionEncoderDecoderWithContextConfig(JointCTCAttentionEncoderDecoderConfig):
#     model_type = "joint-aed-ctc-speech-encoder-decoder-with-context"
#     is_composition = True
#
#
# class JointCTCAttentionEncoderDecoderWithContext(JointCTCAttentionEncoderDecoder):
#     """Custom model for CTC+Attention loss"""
#
#     config_class = JointCTCAttentionEncoderDecoderWithContextConfig
#     base_model_prefix = "joint-aed-ctc-speech-encoder-decoder-with-context"
#     main_input_name = "inputs"
#     supports_gradient_checkpointing = True
#
#     # def __init__(self,
#     #              config: Optional[PretrainedConfig] = None,
#     #              encoder: Optional[PreTrainedModel] = None,
#     #              decoder: Optional[PreTrainedModel] = None):
#     #     if config is None and (encoder is None or decoder is None):
#     #         raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
#     #     if config is None:
#     #         config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
#     #     else:
#     #         if not isinstance(config, self.config_class):
#     #             raise ValueError(f"Config: {config} has to be of type {self.config_class}")
#     #
#     #     if config.decoder.cross_attention_hidden_size is not None:
#     #         if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
#     #             raise ValueError(
#     #                 "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
#     #                 f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
#     #                 f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
#     #                 " `config.encoder.hidden_size`."
#     #             )
#     #
#     #         # initialize with config
#     #         # make sure input & output embeddings is not tied
#     #     config.tie_word_embeddings = False
#     #     super().__init__(config)
#     #
#     #     if encoder is None:
#     #         encoder = Wav2Vec2WithContextForCTC(config.encoder)
#     #
#     #     if decoder is None:
#     #         decoder = GPT2WithContextLMHeadModel(config.decoder)
#     #
#     #     self.encoder = encoder
#     #     self.decoder = decoder
#     #
#     #     if self.encoder.config.to_dict() != self.config.encoder.to_dict():
#     #         logger.warning(
#     #             f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
#     #             f" {self.config.encoder}"
#     #         )
#     #     if self.decoder.config.to_dict() != self.config.decoder.to_dict():
#     #         logger.warning(
#     #             f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
#     #             f" {self.config.decoder}"
#     #         )
#     #
#     #     # make sure that the individual model's config refers to the shared config
#     #     # so that the updates to the config will be synced
#     #     self.encoder.config = self.config.encoder
#     #     self.decoder.config = self.config.decoder
#     #
#     #     # get encoder output hidden size
#     #     self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
#     #     if (
#     #             self.encoder_output_dim != self.decoder.config.hidden_size
#     #             and self.decoder.config.cross_attention_hidden_size is None
#     #     ):
#     #         # encoder outputs might need to be projected to different dimension for decoder
#     #         self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
#     #
#     #     if self.encoder.get_output_embeddings() is not None:
#     #         raise ValueError(
#     #             f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
#     #         )
#     #     self.enc_loss_weight = config.ctc_weight
#     #     self.dec_loss_weight = 1 - config.ctc_weight
#     #     self.lsm_factor = config.lsm_factor
#     #
#     # @classmethod
#     # def from_encoder_decoder_pretrained(
#     #         cls,
#     #         encoder_pretrained_model_name_or_path: str = None,
#     #         decoder_pretrained_model_name_or_path: str = None,
#     #         spec_augment_cfg=None, reverb_cfg=None,
#     #         *model_args,
#     #         **kwargs
#     # ) -> PreTrainedModel:
#     #
#     #     kwargs_encoder = {
#     #         argument[len("encoder_"):]: value for argument, value in kwargs.items() if
#     #         argument.startswith("encoder_")
#     #     }
#     #
#     #     kwargs_decoder = {
#     #         argument[len("decoder_"):]: value for argument, value in kwargs.items() if
#     #         argument.startswith("decoder_")
#     #     }
#     #
#     #     # remove encoder, decoder kwargs from kwargs
#     #     for key in kwargs_encoder.keys():
#     #         del kwargs["encoder_" + key]
#     #     for key in kwargs_decoder.keys():
#     #         del kwargs["decoder_" + key]
#     #
#     #     # Load and initialize the encoder and decoder
#     #     # The distinction between encoder and decoder at the model level is made
#     #     # by the value of the flag `is_decoder` that we need to set correctly.
#     #     encoder = kwargs_encoder.pop("model", None)
#     #     if encoder is None:
#     #         if encoder_pretrained_model_name_or_path is None:
#     #             raise ValueError(
#     #                 "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
#     #                 "to be defined."
#     #             )
#     #
#     #         if "config" not in kwargs_encoder:
#     #             encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
#     #                 encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
#     #             )
#     #             encoder_config.memory_cells = kwargs_encoder.pop("memory_cells")
#     #             encoder_config.memory_dim = kwargs_encoder.pop("memory_dim")
#     #
#     #             encoder_config.architectures = ["Wav2Vec2ForCTC"]
#     #
#     #             if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
#     #                 logger.info(
#     #                     f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
#     #                     "from a decoder model. Cross-attention and casual mask are disabled."
#     #                 )
#     #                 encoder_config.is_decoder = False
#     #                 encoder_config.add_cross_attention = False
#     #
#     #             kwargs_encoder["config"] = encoder_config
#     #
#     #         encoder = Wav2Vec2WithContextForCTC.from_pretrained(encoder_pretrained_model_name_or_path,
#     #                                                             *model_args,
#     #                                                             **kwargs_encoder)
#     #
#     #     decoder = kwargs_decoder.pop("model", None)
#     #     if decoder is None:
#     #         if decoder_pretrained_model_name_or_path is None:
#     #             raise ValueError(
#     #                 "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
#     #                 "to be defined."
#     #             )
#     #
#     #         if "config" not in kwargs_decoder:
#     #             decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
#     #                 decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
#     #             )
#     #             decoder_config.memory_cells = kwargs_decoder.pop("memory_cells")
#     #             decoder_config.memory_dim = kwargs_decoder.pop("memory_dim")
#     #
#     #             if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
#     #                 logger.info(
#     #                     f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
#     #                     f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
#     #                     f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
#     #                 )
#     #                 decoder_config.is_decoder = True
#     #                 decoder_config.add_cross_attention = True
#     #
#     #             kwargs_decoder["config"] = decoder_config
#     #
#     #         if kwargs_decoder["config"].is_decoder is False or kwargs_decoder[
#     #             "config"].add_cross_attention is False:
#     #             logger.warning(
#     #                 f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
#     #                 f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
#     #                 "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
#     #                 "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
#     #                 "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
#     #             )
#     #
#     #         decoder = GPT2WithContextLMHeadModel.from_pretrained(decoder_pretrained_model_name_or_path,
#     #                                                              **kwargs_decoder)
#     #
#     #     # instantiate config with corresponding kwargs
#     #     config = JointCTCAttentionEncoderDecoderWithContextConfig.from_encoder_decoder_configs(encoder.config,
#     #                                                                                            decoder.config, **kwargs)
#     #
#     #     # make sure input & output embeddings is not tied
#     #     config.tie_word_embeddings = False
#     #     return cls(encoder=encoder, decoder=decoder, config=config)
#
# def activate_memory_params(self):
#     self.encoder.activate_memory_params()
#     self.decoder.activate_memory_params()
#
# def freeze(self):
#     for param in self.parameters():
#         param.requires_grad = False
#
# def connect_context_container(self, context_container):
#     self.encoder.connect_context_container(context_container)
#     self.decoder.connect_context_container(context_container)
