# Eurelis-Langchain-SolR-VectorStore

![Python : 11](https://img.shields.io/badge/Python-=3.11-green)
![Code style : black](https://img.shields.io/badge/Code_style-black-black)
![Linting : pylint](https://img.shields.io/badge/Linting-pylint-yellowgreen)

This library allows to use a Solr based vector store with the python version of [LangChain](https://www.langchain.com)

## Usage

This library assume you already have a running Solr instance with a configured dense vector field

```xml
<fieldType name="knn_vector" class="solr.DenseVectorField" vectorDimension="768" similarityFunction="euclidean"/>
<field name="vector" type="knn_vector" indexed="true" stored="true"/>
```

Be sure to set a vectorDimension value corresponding to what yor embeddings model provide.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from eurelis_langchain_solr_vectorstore import Solr

embeddings = HuggingFaceEmbeddings()  # you are free to use any embeddings method allowed by langchain

vector_store = Solr(embeddings)  # with default core configuration
```

You can also specify data about the solr instance and core to use:

```python
vector_store = Solr(embeddings, core_kwargs={
    'page_content_field': 'text_t',  # field containing the text content
    'vector_field': 'vector',        # field containing the embeddings of the text content
    'core_name': 'langchain',        # core name
    'url_base': 'http://localhost:8983/solr', # base url to access solr
    'query_handler': 'select', # handler to use to query solr
    'update_handler': 'update', # update handler for solr
    'metadata_fields': [] # additional solr fields to consider as metadata, ie ['id']
})  # with custom default core configuration
```

In the code above you have both the allowed core arguments and the default value.

### Metadata

The Solr based vector store also supports storing and filtering on metadata.

Metadata are mapped into Solr using the following convention: metadata_*{key}*_*{type}* with 
key being the original metadata key, and type is automatically inferred as:

- *i* for integer fields
- *d* for float fields
- *s* for string fields
- *b* for boolean fields

The *vector_search* method take an optional *where* param expecting a dict:

- dict item key: base name of a metadata field
- dict item value: value expected in the metadata field

Example using the vector store as a retriever:

```python
retriever = vector_store.as_retriever()
```

Example adding filter instructions to the retriever
```python
retriever = vector_store.as_retriever(search_kwargs={'filter': {'language': 'en', 'year': 2000}})
```

## Docker

A docker compose file is present in the etc/docker folder, use it with

```bash
docker compose up -d
```

To launch a solr instance with a core named *langchain* and a 'vector' field with 768 dimensions.
