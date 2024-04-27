import torch
import torch.nn as nn
from typing import Union, Tuple, List
from attrdict import AttrDict

from point_cloud.encoders import PointNetEncoder
from vision_language.encoders import MobileVLMVisionEncoder, MobileVLMLanguageEncoder
from cross_attention import CrossAttention


class InstructNav(nn.Module):
    '''
    Implementation of the InstructNav model, which performs socially aware navigation based on language instructions,
    while fusing multi-sensory information from point clouds and images.

    Example config:
    cfg = {
        'obs_dim': 20,
        'pred_dim': 16,
        'lstm_hidden_dim': 10,
        'lstm_num_layers': 4,
        'env_out_dim': 512,
        'lang_out_dim': 512,
        'lin_mlp_shape': [1024, 32],
        'ang_mlp_shape': [1024, 32],
        'num_xttn_heads': 2
    }
    '''
    def __init__(self, 
                 cfg: Union[dict, AttrDict], 
                 trainable: bool = True) -> None:
        super().__init__()
        
        self.trainable = trainable
        self.cfg = self.process_cfg(cfg)

        # for processing point clouds
        self.pc_encoder = PointNetEncoder()
        self.pc_dim = self.pc_encoder.emb_dim
        
        # for processing visual and language context
        self.vision_encoder = MobileVLMVisionEncoder()
        self.lang_encoder = MobileVLMLanguageEncoder()
        self.vision_dim = self.vision_encoder.emb_dim
        self.lang_dim = self.lang_encoder.emb_dim

        # for temporal processing 
        self.env_dim = self.vision_dim + self.pc_dim # 1024 + 2048
        self.lstm = nn.LSTM(self.env_dim, self.cfg.lstm_hidden_dim, self.cfg.lstm_num_layers, batch_first=True)

        # compression
        self.env_out_dim = self.cfg.env_out_dim # 256
        self.lang_out_dim = self.cfg.lang_out_dim # 256
        self.env_fc = nn.Linear(self.cfg.lstm_hidden_dim, self.env_out_dim)
        self.lang_fc = nn.Linear(self.lang_dim, self.lang_out_dim)

        # compressed cross-modal processing
        self.x_attention = CrossAttention(self.lang_out_dim, 
                                          self.env_out_dim, 
                                          self.env_out_dim, 
                                          num_heads=self.cfg.num_xttn_heads)

        # linear and angular velocity prediction
        self.lin_mlp = nn.Sequential(*[nn.Linear(in_features, out_features) 
                                       for in_features, out_features in zip(self.cfg.lin_mlp_shape[:-1], self.cfg.lin_mlp_shape[1:])])
        self.ang_mlp = nn.Sequential(*[nn.Linear(in_features, out_features) 
                                       for in_features, out_features in zip(self.cfg.ang_mlp_shape[:-1], self.cfg.ang_mlp_shape[1:])])
        
        # navigation instruction embedding, computed and stored here,
        # since instructions are not available at all timesteps
        self.instruct_emb = None


    def process_cfg(self, cfg: Union[dict, AttrDict]) -> AttrDict:
        '''
        Function to check the format of the input configuration 
        and process it into an AttrDict for further use.
        '''
        if isinstance(cfg, dict):
            return AttrDict(cfg)
        elif isinstance(cfg, AttrDict):
            return cfg
        else:
            raise TypeError("Unsupported type for cfg. Must be either dict or AttrDict.")


    def init_lstm(self, batch_size) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to intiialize self.lstm with a zero hidden state.
        '''
        h0 = torch.zeros(self.cfg.lstm_num_layers, batch_size, self.cfg.lstm_hidden_dim)
        c0 = torch.zeros(self.cfg.lstm_num_layers, batch_size, self.cfg.lstm_hidden_dim)
        return h0, c0
    

    def forward_batch(
        self, 
        image: torch.Tensor, 
        pc: torch.Tensor, 
        instruction: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the model with batch processing

        Args:
            image (torch.Tensor): Camera images. shape: (batch, obs_dim, channel, width, height), w = 340, h = 340
            pc (torch.Tensor): Point clouds. shape: (batch, obs_dim, num_points, 3)
            instruction (List[str]): Batch of navigation instructions. shape: (batch, )

        Returns:
            lin_vel, ang_vel: (torch.Tensor, torch.Tensor) linear and angular velocity 
                of the agent across the prediction horizon. shape, both: (batch, pred_dim)
        '''
        # encoding instructions
        instruct_emb = []
        for ins in instruction:
            instruct_emb.append(self.lang_fc(self.lang_encoder(ins)[:, -1, :]))
        instruct_emb = torch.stack(instruct_emb, dim=0) # shape: (batch, self.lang_out_dim)

        # encoding vision and point clouds: building a representation of the environment
        vision_emb = []
        pc_emb = []
        for i in range(len(instruction)):
            vision_emb.append(self.vision_encoder(image[i])) # shape: (obs_dim, self.vision_dim)
            pc_emb.append(self.pc_encoder(pc[i])) # shape: (obs_dim, self.pc_dim)
        vision_emb = torch.stack(vision_emb, dim=0) # shape: (batch, obs_dim, self.vision_dim)
        pc_emb = torch.stack(pc_emb, dim=0) # shape: (batch, obs_dim, self.pc_dim)

        # temporal processing: learning the dynamics of evolution of the environment
        batch_size = pc_emb.size(0)
        h, c = self.init_lstm(batch_size)
        vision_pc_emb = torch.concat((vision_emb.mean(dim=2), pc_emb), dim=2) # shape: (batch, obs_dim, self.vision_dim + self.pc_dim)
        env_emb, _ = self.lstm(vision_pc_emb, (h, c)) # shape: (batch, self.cfg.lstm_hidden_dim)
        env_emb = env_emb[:, -1, :]
        env_emb = self.env_fc(env_emb) # compression, shape: (batch, self.env_out_dim)

        # cross-modal processing: learning relationships between language and environment evolution
        emb = self.x_attention(query=instruct_emb, 
                            key=env_emb.unsqueeze(1), 
                            value=env_emb.unsqueeze(1)).squeeze(1) # unsqueezing for seq_len = 1

        # prediction: finally compare with human navigation
        lin_vel = self.lin_mlp(emb)
        ang_vel = self.ang_mlp(emb)

        return lin_vel, ang_vel

    def forward_single(
        self, 
        image: torch.Tensor, 
        pc: torch.Tensor, 
        instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the model with single inference

        Args:
            image (torch.Tensor): Camera images. shape: (obs_dim, channel, width, height), w = 340, h = 340
            pc (torch.Tensor): Point clouds. shape: (obs_dim, num_points, 3)
            instruction (str): Navigation instruction.

        Returns:
            lin_vel, ang_vel: (torch.Tensor, torch.Tensor) linear and angular velocity 
                of the agent across the prediction horizon. shape, both: (pred_dim)
        '''
        # encoding instruction
        instruct_emb = self.lang_fc(self.lang_encoder(instruction)[:, -1, :]) # compression, shape: (self.lang_out_dim, ), picking the last_hidden_state

        # encoding vision and point clouds: building a representation of the environment
        vision_emb = self.vision_encoder(image) # shape: (obs_dim, self.vision_dim)
        pc_emb = self.pc_encoder(pc) # shape: (obs_dim, self.pc_dim)

        # temporal processing: learning the dynamics of evolution of the environment
        batch_size = 1
        h, c = self.init_lstm(batch_size)
        vision_pc_emb = torch.concat((vision_emb.mean(dim=1), pc_emb), dim=1) # shape: (batch, obs_dim, self.vision_dim + self.pc_dim)
        env_emb, _ = self.lstm(vision_pc_emb.unsqueeze(0), (h, c)) # shape: (1, obs_dim, self.cfg.lstm_hidden_dim)
        env_emb = env_emb[:, -1, :] # last output state 
        env_emb = self.env_fc(env_emb) # compression, shape: (1, self.env_out_dim)

        # cross-modal processing: learning relationships between language and environment evolution
        emb = self.x_attention(query=instruct_emb.unsqueeze(0), 
                            key=env_emb.unsqueeze(0), 
                            value=env_emb.unsqueeze(0)).squeeze(0) # unsqueezing for seq_len = 1

        # prediction: finally compare with human navigation
        lin_vel = self.lin_mlp(emb)
        ang_vel = self.ang_mlp(emb)

        return lin_vel, ang_vel
    


    # legacy code
    # def forward(
    #         self, 
    #         image: torch.Tensor, 
    #         pc: torch.Tensor, 
    #         instruction: Union[str, list[str]] = None
    #     ) -> tuple[torch.Tensor, torch.Tensor]:
    #     '''
    #     Forward pass for the model

    #     Args:
    #         image (torch.Tensor): Camera images. shape: (batch, obs_dim, channel, width, height), w = 340, h = 340
    #         pc (torch.Tensor): Point clouds. shape: (batch, obs_dim, num_points, 3)
    #         instruction: (Union[str, list[str]]): Navigation instruction(s). shape: (batch, )

    #     Returns:
    #         lin_vel, ang_vel: (torch.Tensor, torch.Tensor) linear and angular velocity 
    #             of the agent across the prediction horizon. shape, both: (pred_dim)
    #     '''
    #     batch = True if isinstance(instruction, list) else False

    #     # encoding instruction if available
    #     if instruction is not None:
    #         if batch: # batch of instructions
    #             self.instruct_emb = []
    #             for ins in instruction:
    #                 self.instruct_emb.append(self.lang_encoder(ins))
    #             self.instruct_emb = torch.stack(self.instruct_emb, dim=0)
    #             self.instruct_emb = self.lang_fc(self.instruct_emb) # compression, shape: (batch, self.lang_out_dim)

    #         else: # single instruction
    #             self.instruct_emb = self.lang_encoder(instruction)
    #             self.instruct_emb = self.lang_fc(self.instruct_emb) # compression, shape: (self.lang_out_dim, )

    #     # vision and point cloud encoding: building a representation of the environment
    #     if batch:
    #         vision_emb = []
    #         pc_emb = []
    #         for i in range(len(instruction)): # looping over batch dim as image and pc encoders don't support batch and obs_dim
    #             vision_emb.append(self.vision_encoder(image[i])) # shape: (obs_dim, self.vision_dim)
    #             pc_emb.append(self.pc_encoder(pc[i])) # shape: (obs_dim, self.pc_dim)
    #         vision_emb = torch.stack(vision_emb, dim=0) # shape: (batch, obs_dim, self.vision_dim)
    #         pc_emb = torch.stack(pc_emb, dim=0) # shape: (batch, obs_dim, self.pc_dim)
    #     else:
    #         vision_emb = self.vision_encoder(image) # shape: (obs_dim, self.vision_dim)
    #         pc_emb = self.pc_encoder(pc) # shape: (obs_dim, self.pc_dim)

    #     # temporal processing: learning the dynamics of evolution of the environment
    #     h, c = self.init_lstm(batch_size=pc_emb.size(0))
    #     vision_pc_emb = torch.stack((vision_emb, pc_emb), dim=2) # shape: (batch, obs_dim, self.vision_dim + self.pc_dim)
    #     env_emb, _ = self.lstm(vision_pc_emb, (h, c)) # shape: (batch, self.cfg.lstm_hidden_dim)
    #     env_emb = self.env_fc(env_emb) # compression, shape: (batch, self.env_out_dim)

    #     # cross-modal processing: learning relationships between language and environment evolution
    #     emb = self.x_attention(query=self.instruct_emb.unsqueeze(1), 
    #                            key=env_emb.unsqueeze(1), 
    #                            value=env_emb.unsqueeze(1)).squeeze(1) # unsqueezing for seq_len = 1

    #     # prediction: finally compare with human navigation
    #     lin_vel = self.lin_mlp(emb)
    #     ang_vel = self.ang_mlp(emb)

    #     return lin_vel, ang_vel

