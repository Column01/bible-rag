import argparse
import copy
import glob
import json
import os
import subprocess
import zipfile
from argparse import Namespace
from io import BytesIO

import lmstudio as lms
import numpy as np
import requests
from usearch.index import Index
from bible_rag.version_codes import KNOWN_CODES

# The model used for embedding, loaded when first needed
embed_model = None


def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = lms.embedding_model("text-embedding-nomic-embed-text-v1.5")
    return embed_model


def create_index_and_metadata() -> tuple[Index, list]:
    return Index(ndim=768), []


def sanitize(text: str) -> str:
    new_text = (
        text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    )
    return new_text


def format_book(
    book_name: str, content: dict, translation: str
) -> tuple[list[str], list[dict]]:
    documents = []
    metadatas = []
    for chapter, verses in content.items():
        print(f"Working on {book_name} {chapter}")
        for verse_num, verse_text in verses.items():
            # Format the document for what the nomic-ai/modernbert-embed-text-v1.5 model expects
            verse_text = sanitize(verse_text)
            document = (
                f"search_document: {book_name} {chapter}:{verse_num} {verse_text}"
            )
            documents.append(document)
            metadatas.append(
                {
                    "book": book_name,
                    "chapter": chapter,
                    "verse": verse_num,
                    "text": verse_text,
                    "translation": translation,
                }
            )

    return documents, metadatas


def setup(args: Namespace):
    # Creates the data path directory
    data_path = args.data_path
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Delete existing embeddings if we are not resuming
    if not args.resume:
        try:
            os.rmdir(os.path.join(data_path, "embeddings"))
            os.rmdir(os.path.join(data_path, "resources"))
            print("Cleared existing embeddings")
        except FileNotFoundError:
            pass

    # Make the embeddings directory
    if not os.path.exists(os.path.join(data_path, "embeddings")):
        os.mkdir(os.path.join(data_path, "embeddings"))

    # Make the resources directory
    if not os.path.exists(os.path.join(data_path, "resources")):
        os.mkdir(os.path.join(data_path, "resources"))

    print("Downloading scripture cross references")
    # Download scripture cross references
    resp = requests.get(
        "https://a.openbible.info/data/cross-references.zip",
        headers={"User-Agent": "bible-rag by Column01 on GitHub"},
    )

    # Extract the cross references in memory and write them to disk
    if resp.status_code == 200:
        print("Saving cross references file to memory")
        zip_fp = zipfile.ZipFile(BytesIO(resp.content))
        print("Opening the zip file")
        with zip_fp as zip_file:
            print("Extracting cross references...")
            with zip_file.open("cross_references.txt") as cross_references:
                with open(
                    os.path.join(data_path, "resources", "cross_references.txt"), "w"
                ) as fp:
                    fp.write(cross_references.read().decode("utf-8"))

    else:
        exit(
            f"There was an error when downloading, please try again later. Status Code: {resp.status_code}"
        )

    scraper_cmd = [
        "bible-scraper",
        "--output",
        os.path.join(data_path, "bible_data.json"),
    ]
    if args.resume:
        scraper_cmd.append("--resume")
    # Scrape all versions of scripture using the bible scraper tool installed during setup
    subprocess.run(scraper_cmd)
    # Separate the versions into each individual translation
    subprocess.run(
        [
            "separate-versions",
            "--input",
            os.path.join(data_path, "bible_data.json"),
            "--output",
            os.path.join(data_path, "versions"),
        ]
    )

    print("Loading text embedding model...")
    # Generate embeddings for all translations
    embed_model = get_embed_model()

    print("Creating index...")
    for f_name in glob.glob(f"{data_path}/versions/*.json"):
        version_literal = os.path.split(f_name)[-1].replace(".json", "")
        translation = KNOWN_CODES.get(version_literal, version_literal)
        if args.translation and args.translation != translation:
            continue

        # Only do translations we haven't indexed, should only skip if the user uses --resume
        if not os.path.exists(
            os.path.join(
                data_path, "embeddings", f"{translation}_metadata.json"
            )
        ):
            print(f"Working on translation: {translation}")
            version_index, version_metadata = create_index_and_metadata()

            with open(f_name, "r", encoding="utf-8") as fp:
                bible_json = json.load(fp)
                embeddings = []
                for book, content in bible_json.items():
                    documents, metadatas = format_book(book, content, translation)
                    version_metadata.extend(metadatas)
                    with open(
                        os.path.join(
                            data_path,
                            "embeddings",
                            f"{translation}_metadata.json",
                        ),
                        "w",
                        encoding="utf-8",
                    ) as fp:
                        json.dump(version_metadata, fp, indent=4)

                    print(f"[{translation}] Generating embeddings for Book: {book}")
                    book_embeddings = embed_model.embed(documents)
                    embeddings.extend(book_embeddings)

                stacked = np.stack(embeddings)  # [N, 768]
                # Add the embeddings to the version index
                keys = np.arange(len(embeddings))
                version_index.add(keys, stacked)

                version_index.save(
                    os.path.join(
                        data_path,
                        "embeddings",
                        f"{translation}_index.usearch",
                    )
                )

    print(
        "\nAll set to start querying scripture! Run the program again without the setup flag to search over scripture (bible-rag --help)"
    )


def search(args):
    embed_model = get_embed_model()
    data_path = args.data_path

    query = f"search_query: {args.search}"
    encoded_query = np.array(embed_model.embed(query))

    match_collection = []
    documents = []
    translation = args.translation
    if translation:
        if translation in KNOWN_CODES.keys():
            translation = KNOWN_CODES.get(translation, translation)
        if translation in KNOWN_CODES.values():
            index_path = os.path.join(
                data_path, "embeddings", f"{translation}_index.usearch"
            )
            metadata_path = os.path.join(
                data_path, "embeddings", f"{translation}_metadata.json"
            )

        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        index = Index.restore(index_path)
        for document in index.search(encoded_query, args.n_docs):
            data = metadata[document.key]
            data["key"] = document.key
            data["distance"] = float(document.distance)
            data["translation"] = translation
            documents.append(data)
    else:
        for translation in KNOWN_CODES.values():
            index_path = os.path.join(
                data_path, "embeddings", f"{translation}_index.usearch"
            )
            metadata_path = os.path.join(
                data_path, "embeddings", f"{translation}_metadata.json"
            )
            if os.path.exists(metadata_path) and os.path.exists(index_path):
                print(f"Searching index of translation {translation}")
                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as fp:
                    metadata = json.load(fp)
                index = Index.restore(index_path)
                for document in index.search(encoded_query, args.n_docs):
                    data = metadata[document.key]
                    data["key"] = document.key
                    data["distance"] = float(document.distance)
                    data["translation"] = translation
                    documents.append(data)

    # Collect only the first N docs sorted by distance (should work with every translation and with individual ones)
    sorted_docs = sorted(documents, key=lambda x: x["distance"])[:args.n_docs]
    for doc in sorted_docs:
        print(
            f"({doc["key"]} / {doc["distance"]}) [{doc["translation"]}] {doc["book"]} {doc["chapter"]}:{doc["verse"]} {doc["text"]}"
        )

    if args.output:
        with open("output.json", "w", encoding="utf-8") as fp:
            json.dump(documents, fp, indent=4)


def main():
    parser = argparse.ArgumentParser(
        prog="bible_rag",
        description="A CLI tool for doing RAG over the text of the bible",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="TAKES A LONG TIME! Run when you do not need your computer (overnight). Does the initial setup of the program up for running RAG over scripture.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Use in combination with `--setup` to re-process the data without redownloading all of the scripture",
    )

    parser.add_argument(
        "--data-path",
        help="Sets the path to where Bible data is stored",
        default="data/",
    )

    parser.add_argument(
        "-s",
        "--search",
        help="Searches the Bible for related verses to the entered text",
    )

    parser.add_argument(
        "-t",
        "--translation",
        help="Sets the translation to index or search, otherwise defaults to all translations",
    )

    parser.add_argument(
        "-l",
        "--list-translations",
        help="Lists the translations available",
        action="store_true",
    )

    parser.add_argument(
        "-n", "--n-docs", help="Number of documents to retrieve", type=int, default=5
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Outputs the results to a .json file",
        action="store_true",
    )

    args = parser.parse_args()
    if args.list_translations:
        for name, version_code in KNOWN_CODES.items():
            print(f"{version_code}:")
            print(f"    Name: {name}")
    if args.setup:
        setup(args)

    if args.search:
        search(args)


if __name__ == "__main__":
    main()
