import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
from threading import Thread
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__),"..", ".env"))

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs):
        self.queue.put(None)

class StreamableChain():
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)
        def task():
            self(input, callbacks=[handler])

        thread = Thread(target=task)
        thread.start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token

def StreamChain(StreamableChain, LLMChain):
    pass

chat = ChatOpenAI(streaming=True)
prompt = ChatPromptTemplate.from_messages(
    ("human", "{content}")
)

chain = StreamChain(llm=chat, prompt=prompt)

for chunk in chain.stream(input={"content": "Tell me a joke"}):
    print(chunk)
