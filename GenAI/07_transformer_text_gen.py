from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch, math 
 
def generate_and_evaluate(prompt="Once upon a time", max_length=100): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2") 
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device) 
 
    # Generate sample 
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) 
    output = model.generate( 
        input_ids, 
        max_length=max_length, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.8, 
    ) 
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True) 
 
    # Evaluate perplexity on the prompt 
    with torch.no_grad(): 
        inputs = tokenizer(prompt, return_tensors="pt").to(device) 
        outputs = model(**inputs, labels=inputs["input_ids"]) 
        loss = outputs.loss 
        perplexity = math.exp(loss.item()) 
 
    print("=== Generated Text ===") 
    print(generated_text) 
    print("\n=== Evaluation ===") 
    print(f"Loss: {loss.item():.4f}") 
    print(f"Perplexity: {perplexity:.4f}") 
 
if __name__ == "__main__": 
    generate_and_evaluate("In a distant galaxy,") 