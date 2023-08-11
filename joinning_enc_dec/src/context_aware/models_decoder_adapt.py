from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, PretrainedConfig, \
    SpeechEncoderDecoderConfig, Wav2Vec2ForCTC
from transformers.adapters.composition import adjust_tensors_for_parallel
from transformers.adapters.context import ForwardContext
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, CausalLMOutput, \
    Seq2SeqLMOutput, Wav2Vec2BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import shift_tokens_right
from transformers.utils import (
    logging,
)

from per_utterance.models import JointCTCAttentionEncoderDecoder

logger = logging.get_logger("transformers")


@dataclass
class BaseModelOutputWithPastAndCrossAttentionsWithContext(BaseModelOutputWithPastAndCrossAttentions):
    context_vectors: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None
    encoder_context_vectors: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithContext(CausalLMOutput):
    context_vectors: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Wav2Vec2BaseModelOutputWithContext(Wav2Vec2BaseModelOutput):
    context_vectors: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithContext(BaseModelOutput):
    context_vectors: Optional[Tuple[torch.FloatTensor]] = None


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, 1, H)
        """
        att_w = self.softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1, keepdim=True)

        return utter_rep


class PrevContextCombiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.utterance_pool = SelfAttentionPooling(config.hidden_size)
        self.prev_utterances_attention = nn.MultiheadAttention(config.hidden_size, num_heads=config.num_attention_heads,
                                                               batch_first=True)
        # self.combined_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, context_vectors):
        summary_vector = hidden_states[:, -1, ...]
        if context_vectors is None:
            context_vectors = summary_vector
        else:
            context_vectors = torch.hstack((context_vectors, summary_vector))
        attended_context, _ = self.prev_utterances_attention(hidden_states, context_vectors, context_vectors)
        hidden_states = hidden_states + attended_context
        return hidden_states, context_vectors


class GPT2BlockWithContext(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.context_attention = PrevContextCombiner(config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            context_vectors: Optional[torch.Tensor],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = self.attention_adapters(attn_output, residual, None)

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = self.output_adapters(feed_forward_hidden_states, residual, None)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2ModelWithContext(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2BlockWithContext(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            context_vectors: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        inputs_embeds = self.invertible_adapters_forward(inputs_embeds)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        new_context_vectors = []

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    context_vectors=context_vectors[:, i, ...],
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            new_context_vectors.append(outputs[1])
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)
            # also adjust output shape if necessary
            if getattr(ForwardContext.get_context(), "adapters_parallelized", False):
                output_shape = hidden_states.size()

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        new_context_vectors = torch.stack(new_context_vectors).permute(1, 0, 2, 3)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            context_vectors=new_context_vectors,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class JointCTCAttentionEncoderDecoderWithContext(JointCTCAttentionEncoderDecoder):
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
            decoder = GPT2ModelWithContext.from_config(config.decoder)

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
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if
            argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if
            argument.startswith("decoder_")
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

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder[
                "config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = GPT2ModelWithContext.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
            self,
            inputs: Optional[torch.FloatTensor] = None,
            context_vectors: Optional[torch.Tensor] = None,
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

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if
                          not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if
            argument.startswith("decoder_")
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
                # context_vectors=context_vectors,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                labels=labels,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        # context_vectors = encoder_outputs['context_vectors'] if return_dict else encoder_outputs[-1]
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
            encoder_context_vectors=context_vectors,
            encoder_attentions=encoder_outputs.attentions,
        )
