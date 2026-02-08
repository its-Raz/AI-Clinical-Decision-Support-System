from pinecone import Pinecone, ServerlessSpec
import tiktoken


def get_pinecone_index(PINECONE_API_KEY,PINECONE_INDEX_NAME,embedding_dim):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"\nCreating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Index created")
    else:
        print(f"\nUsing existing Pinecone index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def estimate_tokens_number(text: str):
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))