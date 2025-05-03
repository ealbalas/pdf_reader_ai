import random
import logging
from langchain.chat_models import ChatOpenAI
from app.chat.chains.retrieval import StreamableConvRetrievalChain
from app.chat.models import ChatArgs
from app.chat.vector_store import retriever_map
from app.chat.memories import memory_map
from app.chat.llms import llm_map
from app.web.api import (
    set_conversation_components,
    get_conversation_components,
)
from app.chat.score import random_component_by_score

# Configure logger
logger = logging.getLogger(__name__)

def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    logger.info(f"Building chat for conversation_id: {chat_args.conversation_id}, pdf_id: {chat_args.pdf_id}")

    retriever_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args,
    )
    logger.debug(f"Selected retriever: {retriever_name}")

    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args,
    )
    logger.debug(f"Selected memory: {memory_name}")

    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args,
    )
    logger.debug(f"Selected LLM: {llm_name}")

    set_conversation_components(
        conversation_id=chat_args.conversation_id,
        llm=llm_name,
        memory=memory_name,
        retriever=retriever_name,
    )
    logger.info(f"Set conversation components for conversation_id: {chat_args.conversation_id}")

    condense_question_llm = ChatOpenAI(streaming=False)
    logger.debug("Created condense_question_llm with streaming=False")

    chain = StreamableConvRetrievalChain.from_llm(
        retriever=retriever,
        memory=memory,
        llm=llm,
        condense_question_llm=condense_question_llm,
        metadata=chat_args.metadata,
    )
    logger.info(f"Successfully built StreamableConvRetrievalChain for conversation_id: {chat_args.conversation_id}")
    
    return chain

def select_component(component_name, component_map, chat_args):
    logger.debug(f"Selecting {component_name} for conversation_id: {chat_args.conversation_id}")
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_name]

    if previous_component:
        logger.debug(f"Using previous {component_name}: {previous_component}")
        build_component = component_map[previous_component]
        return previous_component, build_component(chat_args)
    else:
        random_component = random_component_by_score(component_name, component_map)
        logger.debug(f"No previous {component_name} found, randomly selected: {random_component}")
        build_component = component_map[random_component]
        return random_component, build_component(chat_args)
    