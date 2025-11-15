# 2D Turbulence Vision Transformer Emulator

A Vision Transformer-based emulator for 2D turbulence. This project uses masked autoencoders and Vision Transformers to learn and predict spatiotemporal dynamics of 2D turbulent flows.

## Overview

This emulator is designed to:
- Train Vision Transformer models on 2D turbulence data
- Support both single-step and multi-step rollout predictions
- Provide comprehensive evaluation of short- and long-term metrics

## Installation

```bash
# Install required packages

 # 2D turbulence solver package
git clone https://github.com/envfluids/py2d.git
cd py2d && pip install -e ./ 

pip install matplotlib wandb timm einops scipy ruamel.yaml nbformat nbconvert natsort torch torchvision
```