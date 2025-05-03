from langchain.chains import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain
from app.chat.chains.traceable import TraceableChain

class StreamableConvRetrievalChain(TraceableChain, StreamableChain, ConversationalRetrievalChain):
    pass

