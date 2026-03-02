"""arxiv-llm-systems chatbot for querying LLM research papers.

RAG pipeline:
1. Retrieve from ChromaDB (cosine similarity)
2. Rerank with cross-encoder
3. Generate answer with LLM
"""

import chainlit as cl

from chatbot.generator import create_generator
from chatbot.reranker import Reranker
from chatbot.retriever import PaperRetriever
from chatbot.whisper_stt import WhisperTranscriber

# Global components (initialized once)
retriever: PaperRetriever | None = None
reranker: Reranker | None = None
transcriber: WhisperTranscriber | None = None


@cl.on_chat_start
async def on_start():
    """Initialize chat session."""
    global retriever, reranker, transcriber

    if retriever is None:
        await cl.Message(content="Connecting to paper database...").send()
        retriever = PaperRetriever()
        count = retriever._collection.count()
        await cl.Message(content=f"Connected. {count} paper chunks indexed.").send()

    if reranker is None:
        reranker = Reranker()

    if transcriber is None:
        transcriber = WhisperTranscriber()

    generator = create_generator()
    cl.user_session.set("generator", generator)

    await cl.Message(
        content=(
            "Ask me about recent LLM systems research papers.\n"
            "You can also upload an audio file to query by voice."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    query = message.content

    # Handle audio file uploads
    for element in message.elements or []:
        if element.mime and element.mime.startswith("audio/"):
            await cl.Message(content="Transcribing audio...").send()
            query = transcriber.transcribe(element.path)
            await cl.Message(content=f"Transcribed: *{query}*").send()
            break

    if not query.strip():
        await cl.Message(content="Please enter a question or upload an audio file.").send()
        return

    # Step 1: Retrieve (over-fetch for reranking)
    results = retriever.search(query, top_k=10)

    if not results:
        await cl.Message(content="No relevant papers found for your query.").send()
        return

    # Step 2: Rerank with LLM (keep top 3)
    reranked = reranker.rerank(query, results, top_k=3)

    # Step 3: Build context from reranked results
    context_parts = []
    sources = []
    seen_arxiv_ids = set()
    for i, result in enumerate(reranked, 1):
        context_parts.append(
            f"[Paper {i}] {result['title']}\n"
            f"Content: {result['text']}\n"
            f"arXiv: {result.get('arxiv_id', 'N/A')}"
        )
        arxiv_id = result.get("arxiv_id", "")
        if arxiv_id not in seen_arxiv_ids:
            seen_arxiv_ids.add(arxiv_id)
            title = result.get("title", "Untitled")
            sources.append(
                f"- [{title}](https://arxiv.org/abs/{arxiv_id}) "
                f"([PDF](https://arxiv.org/pdf/{arxiv_id}.pdf))"
            )

    context = "\n\n".join(context_parts)

    # Step 4: Generate response with local LLM
    generator = cl.user_session.get("generator")
    prompt = (
        f"Based on the following research papers, answer the user's question.\n\n"
        f"Papers:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a clear, informative answer citing specific papers where relevant."
    )

    response = await cl.make_async(generator.generate)(prompt)

    full_response = f"{response}\n\n**Sources:**\n" + "\n".join(sources)
    await cl.Message(content=full_response).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates (generator config)."""
    generator = create_generator(
        api_base=settings.get("generator_api_base", ""),
        api_key=settings.get("generator_api_key", ""),
        model=settings.get("generator_model", ""),
    )
    cl.user_session.set("generator", generator)
    await cl.Message(content="Generator updated successfully.").send()
