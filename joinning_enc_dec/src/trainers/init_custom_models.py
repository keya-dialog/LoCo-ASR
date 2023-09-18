if __name__ == '__main__':
    train_text = '/mnt/matylda5/xpolok03/projects/LoCo-ASR/text_dump_fisher'

    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "<pad>", "</s>"], vocab_size=5000)

    from tokenizers.pre_tokenizers import Whitespace

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train([train_text], trainer)

    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s>:1 $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    from transformers import PreTrainedTokenizerFast

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token='<pad>'
    )

    wrapped_tokenizer.push_to_hub("fisher_bpe_v2")

    from transformers import Wav2Vec2Config, Wav2Vec2Model

    # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
    configuration = Wav2Vec2Config()
    configuration.num_hidden_layers = 12
    configuration.hidden_size = 512
    configuration.output_hidden_size = 512
    configuration.num_attention_heads = 4
    configuration.num_feat_extract_layers = 3
    configuration.intermediate_size = 2048

    # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    encoder = Wav2Vec2Model(configuration)
    print(encoder.num_parameters())
    encoder.push_to_hub("fisher_enc_12_layers")

    from transformers import Wav2Vec2Config, Wav2Vec2Model

    # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
    configuration = Wav2Vec2Config()
    configuration.num_hidden_layers = 12
    configuration.hidden_size = 512
    configuration.output_hidden_size = 512
    configuration.num_attention_heads = 4
    configuration.num_feat_extract_layers = 7
    configuration.intermediate_size = 2048
    configuration.num_adapter_layers = 2

    # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    encoder = Wav2Vec2Model(configuration)
    print(encoder.num_parameters())
    encoder.push_to_hub("fisher_enc_12_layers_bigger_fe")

    from transformers import Wav2Vec2Config, Wav2Vec2Model

    configuration = Wav2Vec2Config()
    configuration.num_hidden_layers = 12
    configuration.hidden_size = 512
    configuration.output_hidden_size = 512
    configuration.num_attention_heads = 4
    configuration.num_feat_extract_layers = 7
    configuration.intermediate_size = 2048
    configuration.num_adapter_layers = 2
    configuration.feat_extract_norm = "layer"
    configuration.conv_bias = True

    # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    encoder = Wav2Vec2Model(configuration)
    xls_r = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
    encoder.base_model.feature_extractor.load_state_dict(xls_r.base_model.feature_extractor.state_dict())

    print(encoder.num_parameters())

    encoder.push_to_hub("fisher_enc_12_layers_xls_r_fe")

    from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel

    configuration = Wav2Vec2ConformerConfig()
    configuration.num_hidden_layers = 6
    configuration.hidden_size = 512
    configuration.output_hidden_size = 512
    configuration.num_attention_heads = 4
    configuration.num_feat_extract_layers = 3
    configuration.intermediate_size = 2048

    # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    encoder = Wav2Vec2ConformerModel(configuration)
    print(encoder.num_parameters())
    encoder.push_to_hub("fisher_conformer_enc_6_layers")

    from transformers import GPT2Config, GPT2Model

    # Initializing a GPT2 configuration
    configuration = GPT2Config()
    configuration.n_layer = 6
    configuration.vocab_size = 5000
    configuration.hidden_size = 512
    configuration.output_hidden_size = 512
    configuration.n_head = 4

    # Initializing a model (with random weights) from the configuration
    decoder = GPT2Model(configuration)
    print(decoder.num_parameters())

    decoder.push_to_hub("fisher_dec_6_layers")

    from transformers import Speech2TextFeatureExtractor

    config = {
        "num_mel_bins": 84,
        "return_attention_mask": True,
        "sampling_rate": 16000,
    }

    feature_extractor = Speech2TextFeatureExtractor(**config)
    feature_extractor.push_to_hub("fisher_log_mel_extractor")
