import time
import logging
import xml.etree.ElementTree as ET
import os
import shutil
import requests

import gradio as gr
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.CRITICAL)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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

def load_documents(url):
    list_url = url
    if url.endswith(".xml"):
        response = requests.get(url)

        if response.status_code == 200:
            tree = ET.fromstring(response.content)
            loc_elements = tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")

            list_url = [loc_element.text for loc_element in loc_elements]
            print(len(list_url))
        else:
            print("Failed to retrieve sitemap. Status code:", response.status_code)
    loader = WebBaseLoader(list_url)
    documents = loader.load()
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
    global CHROMA_PATH
    CHROMA_PATH_RANDOM = "chroma" + str(time.time())

    db = Chroma.from_documents(
        chunks, embedding=embedding_function, persist_directory=CHROMA_PATH_RANDOM
    )

    db.persist()
    return db

def input_question(question):
    global db
    if not question or not db:
        return "Error: Invalid question or database."

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    prompt = retriever.get_relevant_documents(question)
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

        document_file = gr.Textbox(label="Urls (comma-separated)", lines=2)
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

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            chat, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(debug=True)
