# LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

def get_llm():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cuda:0",
        max_new_tokens=256,
        temperature=0.5,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Document loader
from langchain_community.document_loaders import PyPDFLoader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings
def get_embd_model():
    model_kwargs = {"device": "cuda:0"}
    embd_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs
    )
    return embd_model

# Vector DB
from langchain_chroma import Chroma
def vector_database(chunks):
    embd_model = get_embd_model()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embd_model)
    return vectordb

# Retriever
def retriever(file):
    splits = document_loader(file=file)
    chunks = text_splitter(data=splits)
    vectordb = vector_database(chunks=chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
from langchain.chains import RetrievalQA
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file=file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response["result"]

# Create gradio interface
import gradio as gr
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Simple QA bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7860)