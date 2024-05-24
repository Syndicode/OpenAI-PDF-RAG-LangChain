### OpenAI PDF RAG with LangChain
Ask ChatGPT to answer questions based on your PDF files.

#### How does this work
In this example, we're using GPT-3.5-turbo for inference and the Chroma database for storing embeddings of a PDF file. The PDF file is split into chunks (although it is not necessary in this case because the example file is only 1240 characters long) for embedding and vector storage in Chroma. Then we use LangChain's Retriever to perform a similarity search to facilitate retrieval from Chroma. After this, we ask ChatGPT to answer a question given the context retrieved from Chroma. Finally, we're using the LCEL Runnable protocol to chain together user input, similarity search, prompt construction, passing the prompt to ChatGPT, and parsing the output.

#### How to run
Install dependencies
```bash
pip install langchain langchain-community langchainhub langchain-openai langchain-chroma pypdf
```
Add `OPENAI_API_KEY` to `.env` and run:
```bash
python main.py
```

#### Example output
```
Question: What is the temperature range for E1743 switch?

Answer without RAG:
The temperature range for the E1743 switch is -40°C to 85°C.

Answer with RAG:
The temperature range for the E1743 wireless ON/OFF switch is from 0ºC to 40ºC. It is important not to leave the switch in direct sunlight or near any heat source to prevent overheating. The switch is designed for indoor use only.
```