from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 
from datasets import load_metric 

# Load model and tokenizer (using a summarization-capable model) 
model_name = "facebook/bart-large-cnn" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 

# Initialize Hugging Face pipelines 
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer) 
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer) 

# Example inputs 
text_to_summarize = "The Transformers library by Hugging Face provides state-of-the-art pre-trained models for NLP tasks. It allows you to easily use models for summarization, question answering, and more." 
qa_context = text_to_summarize 
qa_question = "What does the Transformers library provide?" 

# Generate model outputs 
summary = summarizer(text_to_summarize, max_length=50, min_length=10, 
do_sample=False)[0]['summary_text'] 
qa_answer = qa_pipeline(question=qa_question, context=qa_context)['answer'] 
print("Summary:", summary) 
print("QA Answer:", qa_answer) 
rouge = load_metric("rouge") 
reference_summary = ["Hugging Face Transformers provides pre-trained NLP models for summarization and question answering."] 
rouge_score = rouge.compute(predictions=[summary], references=reference_summary) 
print("ROUGE scores:", rouge_score)