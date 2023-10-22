from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from flask import Flask,request, jsonify, Response
from flask_cors import CORS
from flask_sse import sse
import time

from queue import Queue
from threading import Thread

app = Flask(__name__)
CORS(app)

model_path = "models/llama-2-7b-chat.ggmlv3.q2_K.bin"

# callbacks = [StreamingStdOutCallbackHandler()]

token_to_send = ""
message_queue = Queue()

class MyCallbackHandler(BaseCallbackHandler):

    def on_llm_new_token(self, token, **kwargs) -> None:
        #  print every token on a new line
        print(token, end='', flush=True)
        global token_to_send
        token_to_send = token
        message_queue.put(token)


callbacks = [MyCallbackHandler()]
callback_manager = CallbackManager(callbacks)

n_gpu_layers = 15  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
        n_gpu_layers=n_gpu_layers,
        n_batch = n_batch,
        model_path=model_path,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager,
        n_ctx=2000,
        temperature=0.3,
        max_tokens=5000
)


template = """Prompt: You are a summarizer tool. Your task is to condense the content of the text delimited by triple backticks into a detailed summary. Include keypoints in your summary. You are not required to write anything other than the summary itself``{text}```

Summary: """

# template = """Prompt: Your main objective is to act as a personal assitant by answering to the message which is delimited by triple backticks```{text}```

# Response: """



prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def get_llm_response(text):
    result = llm_chain.run(text)
    print(result)




@app.route('/', methods=['GET'])
def test():
    response_data = {
        "result": "Hi, how are you?"
    }
    return response_data
    

@app.route('/stream')
def stream():
    def generate():
        while True:
            token = message_queue.get(block=True)
            yield f'data: {token}\n\n'
    return Response(generate(), mimetype='text/event-stream')


@app.route('/summarize', methods=['POST', 'GET'])
def main():

    print("API hit")

    # loader = TextLoader("source_documents/albert.txt")
    # loadedText = loader.load()

    user_input = None

    if request.method == 'POST':
        user_input = request.json

    # if user_input is None:
    #     return
    

    text = ""
    result = ""

    if request.method == 'POST':
         text = user_input.get('text')


# Callbacks support token-wise streaming
    
    tempText = """Dubbing is a post-production process of re-recording actors’ dialogues, which
is extensively used in filmmaking and video production. It is usually performed
manually by professional voice actors who read lines with proper prosody, and in
synchronization with the pre-recorded videos. In this work, we propose Neural
Dubber, the first neural network model to solve a novel automatic video dubbing
(AVD) task: synthesizing human speech synchronized with the given video from the
text
"""


    # run the inference
    Thread(target=get_llm_response, args=(text,)).start()

    # def generate():
    #     while True:
    #         global token_to_send
    #         if token_to_send != "":
    #             yield f'data: {token_to_send}\n\n'
    #             token_to_send = ""
    #             print("hello")

    # def generate():
    #     while True:
    #         yield f'data: {1}\n\n'
    

    


# start = time.time()
# chain = load_summarize_chain(llm=llm, chain_type="stuff")
# chain.run(loadedText, callbacks=callbacks)
# end = time.time()

# print(f"\n> Answer (took {round(end - start, 2)} s.):")

# prompt = """
# Question: "What is the largest country on Earth?   
# """

# response = llm(prompt)

    response_data = {
        "result": "Loading..."
    }
    return response_data


if __name__ == '__main__':
    app.run(debug=False)