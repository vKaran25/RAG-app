# 📚 RAG-based Document Query Assistant

A **Retrieval-Augmented Generation (RAG)** project built with **LangChain**, **Google Gemini**, and **HuggingFace embeddings**.  
This app allows users to query PDF/CSV documents, retrieve the most relevant content, and generate structured, step-by-step answers.

---

## 🚀 Features

- 🔍 **Document Loaders**: Supports both PDF (`PyPDFLoader`) and CSV (`CSVLoader`) inputs.  
- ✂️ **Text Chunking**: Uses LangChain `CharacterTextSplitter` for efficient document splitting.  
- 🧠 **Embeddings**: Powered by **HuggingFace Sentence Transformers** for semantic vector representation.  
- 🗂️ **Vector Store**: ChromaDB integration for persistent, similarity-based search.  
- 🤖 **LLM Integration**: Google **Gemini 1.5 (flash/pro)** via `langchain_google_genai`.  
- 📑 **Structured Output**: Pydantic schema ensures clean results:  
  - Steps/Syntax for the query  
  - Short Explanation  

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)  
- [Google Gemini API](https://ai.google.dev/) (`langchain_google_genai`)  
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-distilroberta-v1)  
- [Chroma Vector DB](https://www.trychroma.com/)  
- [Pydantic](https://docs.pydantic.dev/latest/)  
- [dotenv](https://pypi.org/project/python-dotenv/)  

---

## ⚙️ Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/rag-query-assistant.git
   cd rag-query-assistant
