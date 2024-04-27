import torch.nn as nn
from .mobilevlm.model.mobilevlm import *
from .mobilevlm.utils import tokenizer_image_token


class MobileVLMVisionEncoder(nn.Module):
    '''
    Frozen vision encoder from MobileVLM with projection
    References:
        https://huggingface.co/mtgv/MobileVLM_V2-1.7B
    '''
    def __init__(self, trainable=True):
        super().__init__()

        self.trainable = trainable
        _, model, _, _ = load_pretrained_model('mtgv/MobileVLM_V2-1.7B',
                                              load_8bit=False,
                                              load_4bit=False,
                                              device_map='cpu',
                                              device='cpu')

        self.vision_tower = model.model.vision_tower
        self.projector = model.model.mm_projector
        self.emb_dim = 2048

        # force device and datatype for cpu
        self.to('cpu').float()
        self.freeze()

    def freeze(self):
        '''
        Function to freeze the parameters of the model
        '''
        for param in self.parameters():
            param.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Module):
                module.requires_grad = False

    def forward(self, image):
        '''
        Forward pass for the model

        Args:
            image (torch.Tensor): shape: (B, C, W, H), W = 340, H = 340
        '''
        emb = self.vision_tower(image)
        emb = self.projector(emb)
        return emb
    

class MobileVLMLanguageEncoder(nn.Module):
    '''
    Frozen language encoder from MobileVLM, basically MobileLLaMA.
    References:
        https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base
    '''
    def __init__(self, trainable=True):
        super().__init__()

        self.trainable = trainable
        self.tokenizer, self.llama, _, _ = load_pretrained_model('mtgv/MobileVLM_V2-1.7B',
                                              load_8bit=False,
                                              load_4bit=False,
                                              device_map='cpu',
                                              device='cpu')

        # override model config for outputting hidden state
        if not self.llama.config.output_hidden_states:
            self.llama.config.output_hidden_states = True

        self.emb_dim = 2048

        # force device and datatype for cpu
        self.to('cpu').float()
        self.freeze()

    def freeze(self):
        '''
        Function to freeze the parameters of the model
        '''
        for param in self.parameters():
            param.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Module):
                module.requires_grad = False

    @torch.no_grad()
    def forward(self, text):
        '''
        Args:
            text (str): input text
        '''
        input_ids = (tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cpu'))
        emb = self.llama.forward(input_ids, output_hidden_states=True).hidden_states[-1]
        return emb