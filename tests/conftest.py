import pytest

from ragroute.cache import KeyTermExtractor
from ragroute.corpus import build_documents
from ragroute.embeddings import HashingEmbedder


@pytest.fixture(scope="session")
def docs():
    return build_documents()


@pytest.fixture(scope="session")
def embedder(docs):
    return HashingEmbedder().fit(d.text for d in docs)


@pytest.fixture(scope="session")
def analyzer(docs):
    return KeyTermExtractor.from_corpus(d.text for d in docs)
