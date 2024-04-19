# Instruct-Nav
We present Instruct-Nav, a model for language-guided socially compliant robot navigation trained on human trajectory preferences in socially crowded and dynamic environments. We leverage the [Multimodal Social Human Navigation Dataset (MuSoHu)](https://cs.gmu.edu/~xiao/Research/MuSoHu/) and build a semantically and temporally consistent language instruction generation pipeline to enable instruction tuning for navigation. The dataset contains approximately 20 hours, 300 trajectories, and 100 kilometers of socially compliant navigation demonstrations collected by 13 human demonstrators in both indoor and outdoor environments. We propose **Instruct-MuSoHu**, which attaches generated instructions to semantically distinct subsets of the existing dataset.

# Instruction Generation
![Instruction generation pipeline](/assets/instruction_generation.png)

We employ GPT-4 API to generate plausible language instructions based on the visual context of the navigation environment. The figure above describes our instruction generation pipeline where we sample equally spaced RGB images from the MuSoHu bags as visual context for querying GPT-4.

# Architecture
![Model architecture](/assets/architecture.png)

Input modalities and encoders:
- **RGB images**: We leverage the `CLIPVisionTower`, trained as a part of the [mtgv/MobileVLM_V2-1.7B](https://huggingface.co/mtgv/MobileVLM_V2-1.7B) model, as our vision encoder.
- **Point clouds**: We employ the [PointNet](https://arxiv.org/abs/1612.00593) encoder.
- **Language instruction**: We use the `MobileLLaMA-1.4B-Base` model as our language encoder, which trained as a part of `mtgv/MobileVLM_V2-1.7B`.
