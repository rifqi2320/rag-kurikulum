from embedding import encode_batch_function
from vectorstore import vectorstore
from langchain_core.documents import Document
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
import os
from tqdm.auto import tqdm

RAW_DATA = "../data/raw"
PROCESSED_DATA = "../data/processed"


def extract_text_from_pdf(
    doc_name, output_name, batch_encode_function, max_length=8191
):
    if not os.path.exists(os.path.join(PROCESSED_DATA, output_name)):
        file_path = os.path.join(RAW_DATA, doc_name)
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
    else:
        df = pd.read_csv(os.path.join(PROCESSED_DATA, output_name))
        pages = [
            Document(page_content=row["text"], metadata=eval(row["metadata"]))
            for _, row in df.iterrows()
        ]
    texts = [page.page_content for page in pages]
    tokenss = batch_encode_function(texts)
    tokens_count = [len(tokens) for tokens in tokenss]
    indexes = [i for i, count in enumerate(tokens_count) if count > max_length]
    if len(indexes) > 0:
        raise ValueError(
            f"Text in pages {', '.join(indexes)} is too long. Max length is {max_length}"
        )
    df = pd.DataFrame.from_records(
        [
            {
                "text": page.page_content,
                "metadata": {
                    **page.metadata,
                    "id": "{}_{}".format(
                        os.path.split(page.metadata["source"])[-1]
                        .split(".")[0]
                        .split("_")[0],
                        page.metadata["page"],
                    ),
                },
            }
            for page in pages
        ],
    )
    df.to_csv(os.path.join(PROCESSED_DATA, output_name), index=False)
    return pages


# FIXME: get_openai_callback() is not tracking embeddings
with get_openai_callback() as cb:
    raw_docs = os.listdir(RAW_DATA)
    for doc in (pbar := tqdm(raw_docs)):
        pbar.set_postfix_str(f"Processing {os.path.split(doc)[-1]}")
        pages = extract_text_from_pdf(doc, f"{doc}.csv", encode_batch_function)
        vectorstore.add_documents(pages, ids=[page.metadata["id"] for page in pages])
    print(cb)
