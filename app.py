import time
import logging
import xml.etree.ElementTree as ET
import os
import shutil
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    OnlinePDFLoader,
)
from langchain.document_loaders import PyPDFLoader
import gradio as gr
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CHROMA_PATH = "chroma"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.CRITICAL)
from huggingface_hub import login

access_token_read = "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"
access_token_write = "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"
login(token=access_token_read)

model_kwargs = {'device': 'cuda'}
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
embedding_function = OpenAIEmbeddings(openai_api_key="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")

PROMPT_TEMPLATE = """
Trả lời câu hỏi dựa trên ngữ cảnh sau đây:

{context}

Trả lời câu hỏi dựa trên ngữ cảnh trên: {question}
"""

db = None  # Initialize db as a global variable


def generate_data_store(url, callback=None):
    global db
    documents = load_documents(url)
    chunks, split_text_into = split_text(documents)
    db = save_to_chroma(chunks)
    if callback:
        callback()
    return db, split_text_into


def build_the_bot(url):
    generate_data_store(url, callback=lambda: None)  # Use a callback to set db
    return "Successfully built the bot!"


def load_documents(documents):
    if type(documents) == str:
        list_url = documents
        if documents.endswith(".xml"):
            response = requests.get(documents)

            if response.status_code == 200:
                tree = ET.fromstring(response.content)
                loc_elements = tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")

                list_url = [loc_element.text for loc_element in loc_elements]
                print(len(list_url))
            else:
                print("Failed to retrieve sitemap. Status code:", response.status_code)
        loader = WebBaseLoader(list_url)
    else:
        loader = OnlinePDFLoader(documents)

    documents = loader.load()
    print(documents)
    return documents


def split_text(documents):
    default_chunk_separators = ['\n\n', '\n', ' ', '']
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=default_chunk_separators,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks, f"Split {len(documents)} documents into {len(chunks)} chunks."


def save_to_chroma(chunks):
    db = Chroma.from_documents(
        chunks, embedding=embedding_function
    )

    return db


def input_question(question):
    global db
    if not question or not db:
        return "Error: Invalid question or database."

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=question)
    print(prompt)
    prompt = generate_response(model, tokenizer, prompt)
    return prompt


def user(user_message, history):
    return "", history + [[user_message, None]]


def chat(history):
    if history[-1][1] is None or history[-1][1] == "":
        bot_message = input_question(history[-1][0])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.005)
            yield history


def generate_response_avahawk(input_text):
    if input_text:
        bot_message = input_question(input_text)
        return bot_message


def generate_response(model, tokenizer, instruction, max_new_tokens=1024, temperature=1.0, top_k=50, top_p=0.9):
    PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"
    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})

    input_ids = tokenizer(input_prompt, return_tensors="pt")

    outputs = model.generate(
        inputs=input_ids["input_ids"].to("cuda"),
        attention_mask=input_ids["attention_mask"].to("cuda"),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = response.split("### Trả lời:")[1]

    return response


model_path = "vinai/PhoGPT-7B5-Instruct"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.init_device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with gr.Blocks() as demo:
    gr.Markdown('<h2 style="font-style: italic;">AvaHawk: Document Chatbot with PhoGPT VinAI RAG Integration</h2>')

    with gr.Tab("Select Document"):
        gr.Markdown('''
                    ## *Solution: What is AvaHawk?*

                    AvaHawk enables you to have interactive conversations with uploaded documents using the powerful capabilities of PhoGPT. 
                    The application intelligently dissects the document into smaller, meaningful sections, employing a robust Deep Averaging Network Encoder to generate embeddings.

                    - **Semantic Search:** A meticulous semantic search is conducted on the document's content, extracting the most pertinent information.

                    - **PhoGPT Integration:** The relevant embeddings are then passed to PhoGPT, leveraging its advanced natural language understanding.

                    - **Custom Logic:** A specially designed logic generates precise and context-aware responses.

                    - **Credible Citations:** The responses even include page numbers in square brackets ([]), enhancing credibility and aiding in quick information retrieval.

                    - **Improved Responses:** The quality of responses surpasses generic ones generated by PhoGPT alone, providing more nuanced and accurate answers.

                    Elevate your document interactions with AvaHawk and experience a new level of conversational depth and intelligence.
                ''')

        # document_file = gr.File(label="Upload your document")
        document_file = gr.Textbox(label="Document url")

        text_output = gr.Textbox(label="Document content")
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(build_the_bot, [document_file], text_output)

    with gr.Tab("AvaHawk"):
        chatbot = gr.Chatbot(avatar_images=["assets/user.png", "assets/2023-12-12 21.32.53.jpg"], height=450)
        msg = gr.Textbox(label="Enter your query here")
        clear = gr.ClearButton([msg, chatbot])

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            chat, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Tab("PhoGPT"):
        chatbot = gr.Chatbot(avatar_images=["assets/user.png", "assets/_wjiAzCq_400x400.jpeg"], height=450)
        msg = gr.Textbox(label="Enter your query here")
        clear = gr.Button("Clear")


        def user(user_message, history):
            return "", history + [[user_message, None]]


        def bot(history):
            bot_message = generate_response(model, tokenizer, history[-1][0])
            history[-1][1] = ""
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.002)
                yield history


        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(debug=True, share=True)
