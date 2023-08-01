class Mapping():
    def __init__(self, model_asr, ckpt):
        self.model_asr = model_asr
        self.ckpt = ckpt
        self.encoder_map, self.encoder_values = self.map_encoder()
        self.encoder_state_dict = self.map_states_encoder()
        self.speech_prenet_map, self.speech_prenet_values = self.map_speech_prenet()
        self.speech_prenet_state_dict = self.map_states_speech_prenet()
        self.text_prenet_map, self.text_prenet_values = self.map_text_prenet()
        self.text_prenet_state_dict = self.map_states_text_prenet()

    def search_mapping(self, base_name, hf_list):
        compat = ""
        #print("BASE:", base_name)
        for l,p in hf_list:
            if "fc1.weight" in base_name and "feed_forward.intermediate_dense.weight" in l and base_name.split('.')[4] == l.split('.')[4]:# and base_name.split('.')[2] == l.split('.')[5]:
                compat = l
                break
            elif "fc1.bias" in base_name and "feed_forward.intermediate_dense.bias" in l and base_name.split('.')[4] == l.split('.')[4]:# and base_name.split('.')[2] == l.split('.')[5]:
                compat = l
                break
            
            elif "fc2.weight" in base_name and "feed_forward.output_dense.weight" in l and base_name.split('.')[4] == l.split('.')[4]:# and base_name.split('.')[2] == l.split('.')[5]:
                compat = l
                break
            elif "fc2.bias" in base_name and "feed_forward.output_dense.bias" in l and base_name.split('.')[4] == l.split('.')[4]:# and base_name.split('.')[2] == l.split('.')[5]:
                compat = l
                break
            elif l.startswith(base_name):
                compat = l
                break
        return compat
    
    def map_encoder(self):
        new_hf_keys_list = [(name, p.size()) for name, p in self.model_asr.named_parameters() if name.startswith("speecht5.encoder.wrapped_encoder")]
        
        encoder_mapping = {}

        hf_roi=3
        bs_roi = 1
        encoder_name = "encoder"
        encoder_hf = "speecht5.encoder.wrapped_encoder"
        for name, p in self.ckpt['model'].items():
            if name.startswith(encoder_name):
                splitt = name.split('.')[bs_roi]
                if splitt == "pos_emb":
                    hf_name = self.search_mapping(encoder_hf+"."+"embed_positions", new_hf_keys_list)
                    encoder_mapping[hf_name] = name
                    
                elif splitt == "layer_norm":
                    for norm,_ in new_hf_keys_list:
                        common = '.'.join(norm.split('.')[hf_roi:])
                        if common in name:
                            encoder_mapping[norm] = name
                            
                elif splitt == "layers":
                    layer_num = name.split('.')[bs_roi+1]
                    layer_type = name.split('.')[bs_roi+2]
                    if layer_type == "self_attn":
                        layer_type = "attention"
                    elif layer_type == "self_attn_layer_norm":
                        layer_type = "layer_norm"
                    
                    if len(name.split('.')) <= bs_roi+4:
                        #fix fc1 here
                        layer_value = name.split('.')[bs_roi+3]
                        #print("################", layer_value)
                        hf_name = self.search_mapping('.'.join([encoder_hf,splitt,layer_num,layer_type,layer_value]), new_hf_keys_list)
                        if len(hf_name) == 0:
                            encoder_mapping[name] = None
                        else:
                            encoder_mapping[hf_name] = name
                    else:
                        layer_value = name.split('.')[bs_roi+3]
                        layer_weight = name.split('.')[bs_roi+4]
                        hf_name = self.search_mapping('.'.join([encoder_hf,splitt,layer_num,layer_type,layer_value,layer_weight]), new_hf_keys_list)
                        if len(hf_name) == 0:
                            encoder_mapping[name] = None
                        else:
                            encoder_mapping[hf_name] = name
                else:
                    continue
                    
        #RESULT
        new_dict_state = {}
        for hf_layer, bs_layer in encoder_mapping.items():
            for name, p in self.ckpt['model'].items():
                if bs_layer == name:
                    new_dict_state[hf_layer] = p
                    break

        return new_dict_state, encoder_mapping
    
    def map_states_encoder(self):
        new_state_dict = {}
        for layer, param in self.encoder_map.items():
            new_layer = layer.split('.',maxsplit=3)[-1]
            new_state_dict[new_layer] = param
        return new_state_dict
    
    def map_speech_prenet(self):
        prenet_mapping = {}

        hf_roi = 2
        bs_roi = 0
        for name, p in self.model_asr.named_parameters():
            if name.startswith("speecht5.encoder.prenet"):
                #print(name, p.shape)
                splitt = name.split('.')
                for name_base, p_base in self.ckpt['model'].items():
                    if name_base.startswith("speech_encoder_prenet"):
                        splitt_base = name_base.split('.')
                        
                        if splitt[hf_roi+1] == "masked_spec_embed" and splitt_base[bs_roi+1] == "mask_emb":
                            prenet_mapping[name] = name_base
                            break
                        elif "feature_projection.layer_norm.weight" in name and "layer_norm.weight" in name_base:
                            prenet_mapping[name] = name_base
                            break
                        elif "feature_projection.layer_norm.bias" in name and "layer_norm.bias" in name_base:
                            prenet_mapping[name] = name_base
                            break
                        elif "feature_projection.projection.weight" in name and "post_extract_proj.weight" in name_base:
                            prenet_mapping[name] = name_base
                            break
                        elif "feature_projection.projection.bias" in name and "post_extract_proj.bias" in name_base:
                            prenet_mapping[name] = name_base
                            break
                        elif "feature_encoder" in name and "feature_extractor" in name_base:
                            #print("Original", name_base)
                            encoder_name = "feature_encoder"
                            conv = "conv"
                            layer_norm = "layer_norm"
                            new_name_base = name_base.replace("feature_extractor", encoder_name).replace("0.weight","conv.weight").replace("2.weight", "layer_norm.weight").replace("2.bias","layer_norm.bias")
                            #print("New:", new_name_base)
                            #print(f"Splits {name.split('.',maxsplit=hf_roi+2)[-1]} =?= {new_name_base.split('.',maxsplit=hf_roi)[-1]}")
                            if name.split(".",maxsplit=hf_roi+2)[-1] == new_name_base.split(".",maxsplit=hf_roi)[-1]:
                                prenet_mapping[name] = name_base
                                break
                        elif "pos_conv" in name and "pos_conv" in name_base:
                            new_name_base = name_base.replace("pos_conv", "pos_conv_embed").replace("0", "conv")
                            #print(f"Splits {name.split('.',maxsplit=hf_roi+1)[-1]} =?= {new_name_base.split('.',maxsplit=hf_roi-1)[-1]}")
                            if name.split(".",maxsplit=hf_roi+1)[-1] == new_name_base.split(".",maxsplit=hf_roi-1)[-1]:
                                #print("HERE")
                                prenet_mapping[name] = name_base
                                break

        prenet_dict_state = {}
        for hf_layer, bs_layer in prenet_mapping.items():
            for name, p in self.ckpt['model'].items():
                if bs_layer == name:
                    prenet_dict_state[hf_layer] = p
                    break

        return prenet_dict_state, prenet_mapping
    
    def map_states_speech_prenet(self):
        speech_prenet_dict_state = {}
        for layer, param in self.speech_prenet_map.items():
            new_layer = layer.split('.',maxsplit=3)[-1]
            #print(new_layer)
            speech_prenet_dict_state[new_layer] = param
            
        for name, param in self.model_asr.named_parameters():
            if "pos_sinusoidal_embed.weights" in name:
                speech_prenet_dict_state[name.split('.',maxsplit=3)[-1]] = param
        
        return speech_prenet_dict_state

    def map_text_prenet(self):
        #Because it is only 2 layers, I am doing it manually
        prenet_mapping = {}
        prenet_mapping["speecht5.encoder.prenet.embed_tokens.weight"] = "text_encoder_prenet.encoder_prenet.0.weight"
        prenet_mapping["speecht5.encoder.prenet.encode_positions.alpha"] = "text_encoder_prenet.encoder_prenet.1.alpha"

        prenet_dict_state = {}
        for hf_layer, bs_layer in prenet_mapping.items():
            for name, p in self.ckpt['model'].items():
                if bs_layer == name:
                    prenet_dict_state[hf_layer] = p
                    break

        return prenet_dict_state, prenet_mapping
    
    def map_states_text_prenet(self):
        text_prenet_dict_state = {}
        for layer, param in self.text_prenet_map.items():
            new_layer = layer.split('.',maxsplit=3)[-1]
            text_prenet_dict_state[new_layer] = param
            
        return text_prenet_dict_state
