# pip install torch transformers accelerate 
import torch 
from transformers import DistilBertTokenizer, DistilBertForMaskedLM 
import time 
import copy 
import os 
import psutil 
# --------------------------- 
# Function to measure memory 
# --------------------------- 
def get_memory_usage_mb(): 
    process = psutil.Process(os.getpid()) 
    mem = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB 
    return mem 
# --------------------------- 
# Load pre-trained model 
# --------------------------- 
model_name = "distilbert-base-uncased" 
tokenizer = DistilBertTokenizer.from_pretrained(model_name) 
model = DistilBertForMaskedLM.from_pretrained(model_name) 
model.eval() 
# Sample input x
text = "The quick brown [MASK] jumps over the lazy dog." 
inputs = tokenizer(text, return_tensors="pt") 
# --------------------------- 
# Function to measure inference speed and memory 
# --------------------------- 
def measure_performance(model, inputs, runs=20): 
# Warm-up 
    with torch.no_grad(): 
        _ = model(**inputs) 
    start_mem = get_memory_usage_mb() 
    start_time = time.time() 
    with torch.no_grad(): 
        for _ in range(runs): 
            _ = model(**inputs) 
    end_time = time.time() 
    end_mem = get_memory_usage_mb() 
    avg_time = (end_time - start_time) / runs 
    mem_usage = end_mem - start_mem 
    return avg_time, mem_usage 
# --------------------------- 
# Baseline performance 
# --------------------------- 
baseline_time, baseline_mem = measure_performance(model, inputs) 
print(f"Baseline - Inference time: {baseline_time*1000:.2f} ms, Memory change: {baseline_mem:.2f} MB") 
# --------------------------- 
# 1. Pruning (simple magnitude pruning) 
# --------------------------- 
def prune_model_layerwise(model, amount=0.3): 
    """ 
    Layer-wise magnitude pruning: zeros out the smallest 'amount' weights 
    per layer. 
    """ 
    for name, param in model.named_parameters(): 
        if 'weight' in name and param.dim() > 1:  # prune only linear layers 
            tensor = param.data 
            k = int(tensor.numel() * amount) 
            if k == 0: 
                continue 
            # Flatten, get top-k absolute values 
            flat_tensor = tensor.abs().view(-1) 
            threshold, _ = torch.kthvalue(flat_tensor, k) 
            tensor[tensor.abs() < threshold] = 0.0 
    return model 

# Apply layer-wise pruning 
pruned_model = copy.deepcopy(model) 
pruned_model = prune_model_layerwise(pruned_model, amount=0.3) 
pruned_time, pruned_mem = measure_performance(pruned_model, inputs) 
print(f"Pruned - Inference time: {pruned_time*1000:.2f} ms, Memory change: {pruned_mem:.2f} MB") 
# --------------------------- 
# 2. Quantization (dynamic quantization) 
# --------------------------- 
quantized_model = torch.quantization.quantize_dynamic( 
    copy.deepcopy(model),  
    {torch.nn.Linear},  # quantize only Linear layers 
    dtype=torch.qint8 
) 
quant_time, quant_mem = measure_performance(quantized_model, inputs) 
print(f"Quantized - Inference time: {quant_time*1000:.2f} ms, Memory change: {quant_mem:.2f} MB")