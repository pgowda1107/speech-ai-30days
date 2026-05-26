#!/bin/bash
# ============================================================
# Parakeet Fine-tuning — Based on official nvidia-riva/tutorials
# asr-finetune-parakeet-nemo.ipynb
# Adapted for 2× L40S GPU
# ============================================================
# Official notebooks:
# Fine-tune: github.com/nvidia-riva/tutorials/blob/main/asr-finetune-parakeet-nemo.ipynb
# Deploy:    github.com/nvidia-riva/tutorials/blob/main/asr-deploy-parakeet-ctc.ipynb
# Multi-GPU: github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb
# ============================================================

# ── PRE-FLIGHT: Verify 2× L40S ──────────────────────────────
nvidia-smi
# Confirm: 2× L40S, driver 535+, CUDA 12.x

# Set default NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

# ── STEP 1: Pull NeMo 26.02 container ───────────────────────
# Latest stable for Speech AI (ASR + TTS)
docker pull nvcr.io/nvidia/nemo:26.02

# ── STEP 2: Launch container with both L40S ──────────────────
docker run --gpus all -it --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
  -v /path/to/checkpoints:/checkpoints \
  -v /path/to/tutorials:/tutorials \
  nvcr.io/nvidia/nemo:26.02

# ── Inside container ─────────────────────────────────────────

# Verify both GPUs
python -c "
import torch
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} — {props.total_memory/1e9:.0f}GB VRAM')
"
# Expected:
# GPUs: 2
# GPU 0: NVIDIA L40S — 47GB VRAM
# GPU 1: NVIDIA L40S — 47GB VRAM

# Clone official RIVA tutorials + NeMo
cd /workspace
git clone https://github.com/nvidia-riva/tutorials.git
git clone https://github.com/NVIDIA-NeMo/NeMo.git
export NEMO_DIR=/workspace/NeMo

# ── STEP 3: Download AN4 dataset (from official tutorial) ────
# Exactly as in asr-finetune-parakeet-nemo.ipynb

mkdir -p /data/an4 && cd /data/an4

wget https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz
tar -xvzf an4_sphere.tar.gz

python $NEMO_DIR/scripts/dataset_processing/get_an4_data.py \
  --data_root /data/an4

export DATA_DIR=/data/an4/an4_converted
echo "Train samples: $(wc -l < $DATA_DIR/train_manifest.json)"
echo "Test samples:  $(wc -l < $DATA_DIR/test_manifest.json)"

# ── STEP 4: Tokenizer decision (from official tutorial) ──────
# Under 50h → use pretrained tokenizer as-is (AN4 is tiny — skip tokenizer training)
# Over 50h or new vocab → uncomment and run:
# python $NEMO_DIR/scripts/tokenizers/process_asr_text_tokenizer.py \
#   --manifest=$DATA_DIR/train_manifest.json \
#   --data_root=/data/tokenizer \
#   --vocab_size=128 \
#   --tokenizer=spe \
#   --spe_type=unigram

# ── STEP 5: Evaluate BASE model WER ──────────────────────────
python -c "
import nemo.collections.asr as nemo_asr
import json
from jiwer import wer as compute_wer

model = nemo_asr.models.ASRModel.from_pretrained(
    'stt_en_fastconformer_hybrid_large_pc'
)
model.eval()

audio_paths, refs = [], []
with open('/data/an4/an4_converted/test_manifest.json') as f:
    for line in f:
        item = json.loads(line.strip())
        audio_paths.append(item['audio_filepath'])
        refs.append(item['text'])

hyps = model.transcribe(audio_paths, batch_size=16)
if isinstance(hyps[0], tuple):
    hyps = [h[0] for h in hyps]

score = compute_wer(refs, hyps)
print(f'BASE WER: {score*100:.2f}%')
print(f'Example — REF:  {refs[0]}')
print(f'Example — HYP:  {hyps[0]}')
"

# ── STEP 6: Fine-tune Parakeet on 2× L40S ────────────────────
# Based on exact command from asr-finetune-parakeet-nemo.ipynb
# Multi-GPU additions: devices=2 + strategy=ddp

python $NEMO_DIR/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/hybrid_transducer_ctc/" \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  +init_from_pretrained_model=stt_en_fastconformer_hybrid_large_pc \
  ++model.train_ds.manifest_filepath="$DATA_DIR/train_manifest.json" \
  ++model.validation_ds.manifest_filepath="$DATA_DIR/test_manifest.json" \
  ++model.train_ds.batch_size=16 \
  ++model.validation_ds.batch_size=16 \
  ++model.optim.sched.d_model=1024 \
  ++trainer.devices=2 \
  ++trainer.accelerator=gpu \
  ++trainer.strategy=ddp \
  ++trainer.max_epochs=50 \
  ++trainer.precision=bf16 \
  ++model.optim.name="adamw" \
  ++model.optim.lr=0.1 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.warmup_steps=100 \
  ++exp_manager.exp_dir=/checkpoints/parakeet_an4 \
  ++exp_manager.create_wandb_logger=false

# Watch for val_wer in logs — should drop each epoch
# On 2× L40S: ~20-25 min for 50 epochs on AN4

# ── STEP 7: Evaluate FINE-TUNED model WER ────────────────────
python -c "
import nemo.collections.asr as nemo_asr
import json, glob
from jiwer import wer as compute_wer

checkpoints = glob.glob('/checkpoints/parakeet_an4/**/*.nemo', recursive=True)
latest = sorted(checkpoints)[-1]
print(f'Loading: {latest}')

model = nemo_asr.models.ASRModel.restore_from(latest)
model.eval()

audio_paths, refs = [], []
with open('/data/an4/an4_converted/test_manifest.json') as f:
    for line in f:
        item = json.loads(line.strip())
        audio_paths.append(item['audio_filepath'])
        refs.append(item['text'])

hyps = model.transcribe(audio_paths, batch_size=16)
if isinstance(hyps[0], tuple):
    hyps = [h[0] for h in hyps]

score = compute_wer(refs, hyps)
print(f'FINE-TUNED WER: {score*100:.2f}%')
print(f'Example — REF:   {refs[0]}')
print(f'Example — HYP:   {hyps[0]}')
"

# ── STEP 8: Export with nemo2riva (from deploy tutorial) ─────
# Based on asr-deploy-parakeet-ctc.ipynb

pip install nemo2riva

CHECKPOINT=$(ls -t /checkpoints/parakeet_an4/**/*.nemo 2>/dev/null | head -1)
echo "Exporting: $CHECKPOINT"

nemo2riva \
  --out /checkpoints/parakeet_finetuned.riva \
  $CHECKPOINT

echo "Exported: /checkpoints/parakeet_finetuned.riva"

# ── STEP 9: Deploy as RIVA NIM ───────────────────────────────
# Based on asr-deploy-parakeet-ctc.ipynb
# Exit NeMo container first

exit

# Pull + run RIVA NIM
docker pull nvcr.io/nim/nvidia/parakeet-ctc-1-1b-asr:latest

docker run -it --rm \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 50051:50051 \
  -e NIM_HTTP_API_PORT=8000 \
  -e NIM_GRPC_API_PORT=50051 \
  -v /checkpoints:/opt/riva/models \
  nvcr.io/nim/nvidia/parakeet-ctc-1-1b-asr:latest

# ── STEP 10: Test gRPC inference ─────────────────────────────

pip install nvidia-riva-client

python -c "
import riva.client
import soundfile as sf
import numpy as np

auth = riva.client.Auth(uri='localhost:50051')
asr = riva.client.ASRService(auth)

audio, sr = sf.read('/data/an4/an4_converted/wavs/an268-mbmg-b.wav')
audio_bytes = (audio * 32768).astype(np.int16).tobytes()

config = riva.client.RecognitionConfig(
    language_code='en-US',
    max_alternatives=1,
    enable_automatic_punctuation=True,
    audio_channel_count=1,
    sample_rate_hertz=16000,
    encoding=riva.client.AudioEncoding.LINEAR_PCM
)

response = asr.offline_recognize(audio_bytes, config)
print('TRANSCRIPT:', response.results[0].alternatives[0].transcript)
print('CONFIDENCE:', response.results[0].alternatives[0].confidence)
"

# ============================================================
# SCREENSHOT CHECKLIST FOR LINKEDIN POST
# ============================================================
# 1. nvidia-smi — 2× L40S confirmed
# 2. Training logs — val_wer improving each epoch
# 3. BASE WER vs FINE-TUNED WER side by side
# 4. RIVA NIM serving + gRPC transcript
# ============================================================
