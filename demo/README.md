---
title: Aether Diffusion LM
emoji: 🌀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Aether — masked diffusion language model

Interactive demo of a masked (absorbing-state) diffusion LM. Watch text resolve
out of noise and adjust the compute/quality tradeoff with the NFE slider.

- Code: https://github.com/ameyg910/aether
- Docs: https://ameyg910.github.io/aether/
- Weights: https://huggingface.co/ameyg910/aether-55m

## Deploying

```bash
huggingface-cli repo create aether-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/ameyg910/aether-demo
cp demo/app.py demo/requirements.txt demo/README.md aether-demo/
cd aether-demo && git add -A && git commit -m "demo" && git push
```
