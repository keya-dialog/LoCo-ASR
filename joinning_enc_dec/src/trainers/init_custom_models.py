if __name__ == '__main__':
    train_text = '/mnt/matylda5/xpolok03/projects/LoCo-ASR/text_dump_fisher'

    from tokenizers import ByteLevelBPETokenizer

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=[train_text], vocab_size=5000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
    ])

    from transformers import PreTrainedTokenizerFast

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token='<pad>'
    )

    wrapped_tokenizer.push_to_hub("fisher_bpe")

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
