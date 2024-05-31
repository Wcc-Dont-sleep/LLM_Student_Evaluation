from llama_index.core.query_engine import CitationQueryEngine
from llama_index.legacy import LLMPredictor
from langchain import OpenAI
from llama_index.core import Prompt
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext
)
from pprint import pprint
import os.path
import logging
import sys
import os


api_key = "sk-Cd9qsD7Sjk5FLtBlb6fZ5GMRQmkmGfbZxZHxJPyBrIiFXU9V"
base_url = "https://api.chatanywhere.com.cn"

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = base_url

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125",max_tokens=1024,streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# check if storage already exists
PERSIST_DIR = "./storage"
def get_response(message):
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("./data/Test").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # Either way we can now query the index

    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "Context information is below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )
    QA_PROMPT = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

    # query_engine = CitationQueryEngine.from_args(index)

    response = query_engine.query(message)

    return response
