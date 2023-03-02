# finetoolformer 👨🏼‍🔧
Framework for finetunning the ToolFormer-based LM in a few shots manner

## How it Works 
1. Using zero-shot learning figure out what API to use given users' prompt;
2. Build a prompt for the task and figure out the parameters of API call in "ToolFormer" way;
3. Call the API
4. Concatenate everything to return to user;

## Features
- Zero-shot learning for the task of intent classification
- Simple and intuitive integration for new API's
- Can be easily adopted for some chatbots (e.g. 🧩 [Regis](https://chrome.google.com/webstore/detail/regis-ai-chat-assistant-w/lmmpjdfangjdaligcohaohlbpfaookpc))

## Installation

To work with this framework you need to create an .env file in the root of the repo with the following values:

```
HUGGINGFACE_API_TOKEN=api_key
OPEN_AI_API_TOKEN=api_key
```
