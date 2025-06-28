# Parameter-Efficient-Fine-Tuning-of-LLaMA-3.2-on-a-Medical-Chain-of-Thought-Dataset
This project involves the parameter-efficient fine-tuning (PEFT) of the LLaMA 3.2 (3B) model using a medical Chain-of-Thought (CoT) dataset from Hugging Face. 
# LLaMA 3.2 (3B) Fine-Tuning on Medical Chain-of-Thought Dataset

This repository contains my submission for **Task 02 (Category A)** of the ARCH Technologies ML Internship.

## 📌 Task Objective
To fine-tune LLaMA 3.2 (3B) using PEFT (LoRA) on a medical reasoning dataset formatted in Chain-of-Thought (CoT) style, enabling structured medical response generation.

## 📁 Contents
- `llama3_medical_finetune.ipynb` – Fine-tuning notebook using Unsloth
- `wandb_logs_screenshot.png` – Training loss and validation loss logs
- `inference_example.png` – Inference output on a custom medical prompt
- `llama_finetune_report.pdf` – Final written report

## 🧠 Methodology
- Used Hugging Face dataset: [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- Loaded model: [`unsloth/Llama-3.2-3B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)
- Applied PEFT using LoRA with Unsloth
- Tracked training metrics via Weights & Biases (wandb)
- Evaluated performance using ROUGE-L score

## 💡 Inference Instructions
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="your-username/llama-medical-cot", tokenizer="your-username/llama-medical-cot")
input_text = "<think>Patient complains of fever and neck stiffness.</think>\n<response>"
pipe(input_text, max_new_tokens=100)
