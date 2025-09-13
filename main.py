from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-pro
    api_key=api_key,
)

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-distilroberta-v1'
)

class query_op_processing_model(BaseModel):
    steps_or_syntax: str = Field(description="Steps to perform or the syntax to use the query asked")
    very_short_explanation: str = Field(description="Very short explanation of the query")

loader = PyPDFLoader(file_path='The-Ultimate-Python-Handbook.pdf')
# loader = CSVLoader(file_path='AllRace.csv')
docs = loader.load()

# Split documents into chunks
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator=' '
)
result = splitter.split_documents(docs)

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='my_chroma_db',
    collection_name='sample'
)
vector_store.add_documents(result)
query_input = str(input("Enter your query: "))
query_output = vector_store.similarity_search(query_input, k=5)
formatted_query_output = "\n\n".join([d.page_content for d in query_output])

relevant_doc_fetcher_prompt = PromptTemplate(
    template="""
You are given the following documents:
{docs}

Based on these, answer the query: "{query}"

Provide:
1. Steps or syntax to perform the query (proper).
2. A short explanation of what the query means.
""",
    input_variables=['query', 'docs']
)

structured_llm = llm.with_structured_output(query_op_processing_model)
chain = relevant_doc_fetcher_prompt | structured_llm

result_after_fetching_relevant_doc = chain.invoke({
    'query': query_input,
    'docs': formatted_query_output
})

# Print results
print("\nSteps or Syntax:\n", result_after_fetching_relevant_doc.steps_or_syntax)
print("\nShort Explanation:\n", result_after_fetching_relevant_doc.very_short_explanation)
