import dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = PyPDFLoader("ikea_light_switch_manual.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(document)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is the temperature range for E1743 switch?"
print(f"Question: {question}")
print("Answer without RAG:")
print(llm.invoke([question]).content)
print("Answer with RAG:")
print(rag_chain.invoke(question))

vectorstore.delete_collection()
