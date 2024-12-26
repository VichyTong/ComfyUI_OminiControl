# ComfyUI_RH_OminiControl

**ComfyUI_RH_OminiControl** 是一个基于 [OminiControl](https://github.com/Yuanshi9815/OminiControl) 的 ComfyUI 插件。通过分拆 pipeline 加载，该插件能够在 NVIDIA RTX 4090 显卡上高效运行。同时，空间（spatial）及填充（fill）功能也通过 schnell 模型生成，减少采样步数，提高整体效率。

## 特性

- **优化性能**：通过分拆 pipeline 加载，充分利用 RTX 4090 的计算能力。
- **高效生成**：使用 schnell 模型生成 spatial 及 fill，减少采样步数，提升生成效率。
- **易于安装**：依赖 ComfyUI 常用库，通常无需额外安装。
- **灵活配置**：支持自定义模型路径，便于管理和更新模型。

## 安装指南

### 前置条件

- **ComfyUI**：确保已安装并配置好 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。
- **Python 依赖**：通常不需要额外安装库，但建议安装 `diffusers` 版本 0.31.0 以支持 FluxPipeline。

## 下载插件
克隆本插件仓库到 custom_nodes：
```
git clone https://github.com/HM-RunningHub/ComfyUI_RH_OminiControl.git
```

## 模型目录结构：
```
/models/flux
tree
.
├── FLUX.1-schnell
│   ├── ae.safetensors
│   ├── model_index.json
│   ├── README.md
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── schnell_grid.jpeg
│   ├── text_encoder
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── text_encoder_2
│   │   ├── config.json
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── model.safetensors.index.json
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── tokenizer_2
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── transformer
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   │   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   │   ├── diffusion_pytorch_model-00003-of-00003.safetensors
│   │   └── diffusion_pytorch_model.safetensors.index.json
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
└── OminiControl
    ├── depth-anything-small-hf
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── preprocessor_config.json
    │   └── README.md
    ├── experimental
    │   ├── canny.safetensors
    │   ├── coloring.safetensors
    │   ├── deblurring.safetensors
    │   ├── depth.safetensors
    │   ├── fill.safetensors
    │   └── subject.safetensors
    ├── omini
    │   ├── subject_1024_beta.safetensors
    │   └── subject_512.safetensors
    └── README.md

12 directories, 39 files
```
### 根据上面的目录机构下载并放置以下模型：
```
diffusers格式的flux模型，下载地址：https://huggingface.co/black-forest-labs/FLUX.1-schnell
depth-anything-small-hf/ （用于depth识别，下载地址：https://huggingface.co/LiheYoung/depth-anything-small-hf/tree/main）
experimental/ （下载地址：https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental）
omini/ （下载地址：https://huggingface.co/Yuanshi/OminiControl/tree/main/omini）
```
### 运行案例展示
运端一键运行： https://www.runninghub.cn/post/1865085524393500674
![image](https://github.com/user-attachments/assets/2db219bb-957f-4563-9285-d1a62deb77d1)


### 致谢
感谢 Yuanshi9815 及其 OminiControl 项目提供的基础支持。
