from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import DocArrayInMemorySearch
from dotenv import load_dotenv

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time
import os


from constants import CHROMA_SETTINGS

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)


model_path = "models/llama-2-7b-chat.ggmlv3.q2_K.bin"

callbacks = [StreamingStdOutCallbackHandler()]
callback_manager = CallbackManager(callbacks)

n_gpu_layers = 15 # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        model_path=model_path,
        max_tokens=1000,
        callback_manager=callback_manager,
        verbose=False, # Verbose is required to pass to the callback manager,
        temperature=0.5,
        n_ctx=1000
)


loader = TextLoader("demo/text/gpt.txt")
loadedDocuments = loader.load()

# print(loadedDocuments[0].page_content)

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 30)
splits = splitter.split_documents(loadedDocuments)

print(splits)

embeddings_model_name = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name = embeddings_model_name)

vectorStore = Chroma.from_documents(documents=splits,embedding=embeddings)
retriever = vectorStore.as_retriever()


while True:
    question = input("\nEnter your question: ")
    docs = vectorStore.similarity_search(question)

    docsContent = ""

    for doc in docs:
        docsContent+= doc.page_content

    persist_directory = os.environ.get('PERSIST_DIRECTORY')

    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
# db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
# retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

# chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         verbose=True,
# )

    template = """
        Prompt: Your task is to provide an answer to the user's question solely based on the document text provided to you delimited by triple backticks. Make sure that you stick to the provided information only. If you don't find the answer to the question in the text just say that you can not answer this question ```{docsContent}```

        Question: {question}

        Answer: """

    prompt = PromptTemplate(template=template, input_variables=["docsContent", "question"])

    chain = LLMChain(
        llm = llm,
        prompt=prompt,
    )

    chain.acall()

    start = time.time()

    chain.run(docsContent = docsContent, question = question)

    end = time.time()

    print(f"\n> It (took {round(end - start, 2)} s.):")