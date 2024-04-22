# Instruct-Nav

![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![omniverse](https://img.shields.io/badge/NVIDIA-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)
![gpt-4](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)

We present Instruct-Nav, a model for language-guided socially compliant robot navigation trained on human trajectory preferences in socially crowded and dynamic environments. We leverage the [Multimodal Social Human Navigation Dataset (MuSoHu)](https://cs.gmu.edu/~xiao/Research/MuSoHu/) and build a semantically and temporally consistent language instruction generation pipeline to enable instruction tuning for navigation. The dataset contains approximately 20 hours, 300 trajectories, and 100 kilometers of socially compliant navigation demonstrations collected by 13 human demonstrators in both indoor and outdoor environments. We propose **Instruct-MuSoHu**, which attaches generated instructions to semantically distinct subsets of the existing dataset.

# Instruction Generation
![Instruction generation pipeline](/assets/instruction_generation.png)

We employ GPT-4 API to generate plausible language instructions based on the visual context of the navigation environment. The figure above describes our instruction generation pipeline where we sample equally spaced RGB images from the MuSoHu bags as visual context for querying GPT-4.

# Architecture
![Model architecture](/assets/architecture.png)

Input modalities and encoders:
- **RGB images**: We leverage the `CLIPVisionTower`, trained as a part of the [mtgv/MobileVLM_V2-1.7B](https://huggingface.co/mtgv/MobileVLM_V2-1.7B) model, as our vision encoder.
- **Point clouds**: We employ the [PointNet](https://arxiv.org/abs/1612.00593) encoder.
- **Language instruction**: We use the `MobileLLaMA-1.4B-Base` model as our language encoder, which was trained as a part of `mtgv/MobileVLM_V2-1.7B`.

# Evaluation Setup
We leverage [Nvidia Isaac Sim](https://developer.nvidia.com/isaac-sim) for testing and evaluating Instruct-Nav's capabilities. We deploy our model on a CarterV1 robot with a PhysX Lidar and Isaac camera. Here's the high level workflow of the Action Graph implementing Instruct-Nav functionality

![Action graph](/assets/action_graph.png)
