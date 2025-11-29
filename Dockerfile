FROM python:3.10-slim

# set working directory
WORKDIR /app

# upgrade pip & install dependencies
RUN pip install --upgrade pip && \
    pip install \
    "pydantic>=2.7.4,<3.0.0" \
    "numpy==1.26.4" \
    "langchain==0.3.27" \
    "langchain-core==0.3.79" \
    "langchain-openai==0.3.11" \
    "langchain-pinecone==0.2.12" \
    "pinecone-client==4.0.0" \
    "openai>1.0.0" \
    "python-dotenv" \
    "tavily-python" \
    "langchain_core" \
    "langchain_pinecone" \
    "ebooklib" \
    "html2text" \
    "langchain_community" \
    "chromadb" \
    jupyter
# copy your code into the container (optional for now)
COPY . .

# expose jupyter port - good practice for clarity 
EXPOSE 8888

# default command when container runs
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]

