# Tibble-server

This is the repo for the inference server of the [Tibble](https://github.com/code-sparsh/Tibble) project. 

## Tibble
Tibble is your offline privacy-focussed AI assistant tool, which works totally without Internet.

<details>
<summary><b>Problems Tibble solves</b></summary>

Companies around the world are expanding their dependency on AI based tools but they are also worried about the villain of this story and that villain is none other than giants like OpenAI and Microsoft. Corporate customers do not like to share their internal data with third party platforms for their privacy concerns and that's where Private LLMs come in.


We solve this problem of privacy by offering these companies a way for them to install an on-site AI model inference server fine-tuned on their own private data.

</details>


https://github.com/code-sparsh/Tibble-server/assets/85060248/bbec6b54-ced5-420e-b3ca-ae95476de735


## 📕 Requirements

- A good CPU + GPU combination
- NVIDIA CUDA installed
- Minimum 8 GB RAM
- Python3



## 🛫 Quick Setup

```sh
# Clone the repo
git clone https://github.com/code-sparsh/Tibble-server.git

# Navigate to the project directory
cd Tibble-server

# Download the model weights
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin -P models

# Install the dependencies
pip3 install -r requirements.txt

# -> Run the summarization inference
python3 summarize.py

# -> Run the document Q&A inference
python3 documentQA.py
```


## Inference

For inference use either of the two,

- set-up the [Tibble](https://github.com/code-sparsh/Tibble) desktop application (ElectronJS) locally.

    (OR)

- Open `demo/index.html` using any browser (for a quick demo)



## Download the Model (LLaMA 2 - chat)

Link - https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin

You can download any GGML model of your choice. Put the .bin file inside the ``/models`` folder


## Features (To-Do)
- [X] Document Summarization
- [X] Document Q&A
- [ ] Conversational chatbot
- [ ] Grammar / Sentence corrector

