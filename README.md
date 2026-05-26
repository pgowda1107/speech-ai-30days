# Speech AI — 30 Days Learning in Public

Fine-tuning, deployment and production Speech AI scripts.
Following along on LinkedIn by Prakruthi B Gowda | Solution Architect @NVIDIA

## Day 10 — Parakeet-TDT-v3 Fine-tuning on 2× NVIDIA L40S

| | |
|---|---|
| Model | nvidia/parakeet-tdt-0.6b-v3 |
| Hardware | 2× NVIDIA L40S (96GB total) |
| Dataset | AN4 (CMU Speech) |
| Container | NeMo 26.02 |
| Precision | bf16 |
| Strategy | DDP (2 GPUs) |
| Base WER | 21.73% |
| Fine-tuned WER | 0.13% |
| Improvement | 99.4% |
| Training time | ~42 minutes |
| Best epoch | 7 / 50 |

### Script
See `day10_parakeet_finetune_2xL40S.sh`

### References
- [Official Parakeet fine-tuning tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-parakeet-nemo.ipynb)
- [NeMo ASR examples](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/asr)
- [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [RIVA NIM docs](https://docs.nvidia.com/nim/riva/asr/latest/overview.html)
