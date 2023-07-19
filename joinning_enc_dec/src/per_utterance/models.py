from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, \
    SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel, Wav2Vec2ForCTC
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import shift_tokens_right
from transformers.utils import logging

logger = logging.get_logger("transformers")


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None


class JointCTCAttentionEncoderDecoder(SpeechEncoderDecoderModel):
    """Custom model for CTC+Attention loss based on the ESPNet architecture"""

    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True

    def __init__(self,
                 config: Optional[PretrainedConfig] = None,
                 encoder: Optional[PreTrainedModel] = None,
                 decoder: Optional[PreTrainedModel] = None):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

            # initialize with config
            # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = Wav2Vec2ForCTC(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
                self.encoder_output_dim != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )
        self.enc_loss_weight = config.ctc_weight
        self.dec_loss_weight = 1 - config.ctc_weight

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            spec_augment_cfg=None, reverb_cfg=None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                encoder_config.architectures = ["Wav2Vec2ForCTC"]
                # encoder_config.prefix = 'wav2vec2'

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = Wav2Vec2ForCTC.from_pretrained(encoder_pretrained_model_name_or_path,
                                                     *model_args,
                                                     **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
            self,
            inputs: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            input_values: Optional[torch.FloatTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputLosses]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                labels=labels,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs['hidden_states'][-1] if return_dict else encoder_outputs[1][-1]

        # optionally project encoder_hidden_states
        if (
                self.encoder_output_dim != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = enc_loss = dec_loss = None
        if labels is not None:
            dec_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            dec_loss = loss_fct(dec_logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            enc_loss = encoder_outputs.loss if return_dict else encoder_outputs[0]
            loss = self.enc_loss_weight * enc_loss + self.dec_loss_weight * dec_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputLosses(
            loss=loss,
            enc_loss=enc_loss,
            dec_loss=dec_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
