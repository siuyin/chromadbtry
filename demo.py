import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
)

_gemma_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434", model_name="embeddinggemma"
)
_nomic_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434", model_name="nomic-embed-text"
)


def _qry(col, txt):
    res = col.query(query_texts=[txt], n_results=2)
    print(res)


def ephemeral():
    cl = chromadb.Client()
    col = cl.create_collection(name="my_collection")
    col.add(
        ids=["id1", "id2"],
        documents=[
            "This is a document about pineapples.",
            "This is a document about oranges",
        ],
    )

    _qry(col, "This is a query about Hawaii.")
    _qry(col, "This is a query about Florida.")


def _persistent_db_setup(embedding_function=DefaultEmbeddingFunction()):
    cl = chromadb.PersistentClient(path="/tmp/mychroma.db")
    col = cl.create_collection(
        name="my_collection", embedding_function=embedding_function
    )
    # col = cl.create_collection(name="my_collection")
    col.add(
        ids=["id1", "id2"],
        documents=[
            "This is a document about pineapples.",
            "This is a document about oranges",
        ],
    )


def _persistent_db_teardown():
    cl = chromadb.PersistentClient(path="/tmp/mychroma.db")
    cl.delete_collection(name="my_collection")


def _persistent_db_query(txt):
    cl = chromadb.PersistentClient(path="/tmp/mychroma.db")
    col = cl.get_collection(name="my_collection")

    res = col.query(query_texts=[txt], n_results=2)
    print(res)


def persistent():
    _persistent_db_setup(embedding_function=_nomic_ef)
    _persistent_db_query("This is a query document about Hawaii.")
    _persistent_db_query("This is a query document about Florida.")
    _persistent_db_teardown()
