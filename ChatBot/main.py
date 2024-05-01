from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone
from dotenv import load_dotenv
import os


class ChatBot():
  # Loading environment variables from .env file
  load_dotenv()

  # Loading text data from a file using TextLoader
  loader = TextLoader('./dataset.txt',encoding="utf8")
  documents = loader.load()
  
  # Splitting documents into chunks using CharacterTextSplitter
  text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
  docs = text_splitter.split_documents(documents)
  

  # Initializing Hugging Face embeddings 
  embeddings = HuggingFaceEmbeddings()

  # Initializing Pinecone for efficient vector search
  pinecone.init(
      api_key= os.getenv('PINECONE_API_KEY'),
      environment='gcp-starter'
  )

  index_name = "rag-chatbot"

  # Checking if index exists in Pinecone, creating if not exists
  if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

  # Initializing Hugging Face endpoint with specified parameters
  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceEndpoint(
      repo_id=repo_id, temperature= 0.8, top_p= 0.8, top_k= 50, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
  )


  # Defining a prompt template for the chatbot's responses
  template = """
  You are a bot that anwsers student queries regarding higher studies. These Human will ask you a questions about scholarships and student loans. Use following piece of context to answer the question. 
  If you don't know the answer, just say you don't know. 
  You answer with short and concise answer, no longer than 10 sentences.

  Context: {context}
  Question: {question}
  Answer: 

  """

  # Instantiating a PromptTemplate based on the defined template
  prompt = PromptTemplate(template=template, input_variables=["context", "question"])


  # Defining the processing chain for the chatbot
  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )


# Creating an instance of the ChatBot class
bot = ChatBot()

# Getting user input
input = input("Enter your question: ")

# Invoking the processing chain with the user input
result = bot.rag_chain.invoke(input)


print(result)