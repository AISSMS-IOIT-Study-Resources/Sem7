# pip install langchain chromadb sentence-transformers transformers accelerate pypdf bitsandbytes 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA 
from langchain.llms import HuggingFacePipeline 
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 
from langchain.document_loaders import PyPDFLoader 
from langchain.prompts import PromptTemplate 
# ====================================================== 
# 1⃣ Load Material 
# ====================================================== 
pdf_path = "/content/BHIC-105(English).pdf" 
loader = PyPDFLoader(pdf_path) 
documents = loader.load() 
# ====================================================== 
# 2⃣ Split Text 
# ====================================================== 
splitter = RecursiveCharacterTextSplitter(chunk_size=800, 
chunk_overlap=100) 
texts = splitter.split_documents(documents) 
# ====================================================== 
# 3⃣ Embeddings + Chroma 
# ====================================================== 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
db = Chroma.from_documents(texts, embeddings, persist_directory="book_db") 
retriever = db.as_retriever(search_kwargs={"k": 4}) 
# ====================================================== 
# 4⃣ Load Better LLM (choose one) 
# ====================================================== 
model_id = "mistralai/Mistral-7B-Instruct-v0.2" 
tokenizer = AutoTokenizer.from_pretrained(model_id) 
model = AutoModelForCausalLM.from_pretrained( 
model_id, 
torch_dtype="auto", 
device_map="auto" 
) 
pipe = pipeline( 
"text-generation", 
model=model, 
tokenizer=tokenizer, 
max_new_tokens=500, 
temperature=0.3, 
do_sample=True, 
top_p=0.95 
) 
llm = HuggingFacePipeline(pipeline=pipe) 
# ====================================================== 
# Prompt 
# ====================================================== 
prompt_template = """ 
You are an expert in Indian history helping BA students prepare academic 
assignments. 
Use the given context to write a structured answer. 
Question: {question} 
Context: 
{context} 
Write the answer in assignment style: - Start with **Introduction** - Then give **Detailed Explanation** (3–5 paragraphs) - End with **Conclusion** - Use clear, formal, and academic English. - Keep sentences original (not copy-paste). 
Answer: 
""" 
PROMPT = PromptTemplate( 
input_variables=["context", "question"], 
template=prompt_template 
) 
qa_chain = RetrievalQA.from_chain_type( 
llm=llm, 
retriever=retriever, 
chain_type_kwargs={"prompt": PROMPT} 
) 
# ====================================================== 
# 6⃣ Ask Question 
# ====================================================== 
query = "Discuss the powers and functions of the feudatory chiefs" 
answer = qa_chain.run(query) 
print("Question:", query) 
print("Answer:\n") 
print(answer) 