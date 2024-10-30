import json
import langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain.vectorstores import faiss
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader, PyPDFLoader
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr
from functools import partial
from operator import itemgetter

from pprint import pprint

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
    )

#import Llama2 
instruct_llm = ChatOllama(model="llama2", temperature=0.6, num_predict=256)

#Using Nvidia embeddings
embedder= OllamaEmbeddings(model="llama2")

#Building our vectorstore
#convstore= FAISS.from_texts(docs, embedding= embedder)

#Initialize the retriever
#retriever= convstore.as_retriever()



docs=[
    ArxivLoader(query="1706.03762").load(),  ## Attention Is All You Need Paper
    #ArxivLoader(query="1810.04805").load(),  ## BERT Paper
    #ArxivLoader(query="2005.11401").load(),  ## RAG Paper
    #ArxivLoader(query="2205.00445").load(),  ## MRKL Paper
    #ArxivLoader(query="2310.06825").load(),  ## Mistral Paper
    ArxivLoader(query="2306.05685").load(),  ## LLM-as-a-Judge
    
    ]



for doc in docs:
    content=json.dumps(doc[0].page_content)
    if "References" in content:
        doc[0].page_content = content[:content.index("References")]
    

#Chunking the documents and remove very short chunks
print("Start chunking")
doc_chunks=[text_splitter.split_documents(doc) for doc in docs]
doc_chunks=[[c for c in dchunks if len(c.page_content)>200] for dchunks in doc_chunks]


#Adding the big-picture details
Doc_string="Available Documents: "
Doc_metadata=[]
for chunk in doc_chunks:
    metadata= getattr(chunk[0], 'metadata',{})
    Doc_string+= "\n - " + metadata.get('Title')
    Doc_metadata+= [str(metadata)]
    
BP_Chunks=  [Doc_string] + Doc_metadata



## Printing out some summary information for reference
print(Doc_string, '\n')
for i, chunks in enumerate(doc_chunks):
    print(f"Document {i}")
    print(f" - # Chunks: {len(chunks)}")
    print(f" - Metadata: ")
    pprint(chunks[0].metadata)
    print()   
    

#Consctructing the vector store
vecstore=[FAISS.from_texts(BP_Chunks, embedding=embedder)]
vecstore+=[FAISS.from_documents(doc_chunk,embedding=embedder) for  doc_chunk in doc_chunks]  




embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
docstore = aggregate_vstores(vecstore)

print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")


convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')


initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {Doc_string}\n\nHow can I help you?"
)


chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
    "Answer in less than 250 words"
), ('user', '{input}')])


def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))


stream_chain = chat_prompt| RPrint() | instruct_llm | StrOutputParser()

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## Optional; Reorders longer documents to center of output text
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

retrieval_chain = (
    {'input' : (lambda x: x)}
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    ## HINT: Our solution uses RunnableAssign, itemgetter, long_reorder, and docs2str
    | RunnableAssign({'history' : itemgetter('input')| convstore.as_retriever()| long_reorder | docs2str })
    | RunnableAssign({'context' : itemgetter('input')|  docstore.as_retriever()| long_reorder | docs2str})
)

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ## First perform the retrieval based on the input message
    retrieval = retrieval_chain.invoke(message)
    line_buffer = ""

    ## Then, stream the results of the stream_chain
    for token in stream_chain.stream(retrieval):
        buffer += token
        ## If you're using standard print, keep line from getting too long
        yield buffer if return_buffer else token

    ## Lastly, save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)


## Start of Agent Event Loop
test_question = "Tell me about RAG!"  

## Before you launch your gradio interface, make sure your thing works
for response in chat_gen(test_question, return_buffer=False):
    print(response, end='')




chatbot = gr.Chatbot(value = [[None, initial_msg]])
demo = gr.ChatInterface(chat_gen, chatbot=chatbot).queue()

try:
    

    demo.launch(debug=True, share=True, show_api=False)
    
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e

## Save and compress your index
docstore.save_local("docstore_index")
#!tar czvf docstore_index.tgz docstore_index

#!rm -rf docstore_index

#from langchain_community.vectorstores import FAISS

# # embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4", truncate="END")
# !tar xzvf docstore_index.tgz
# new_db = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
# docs = new_db.similarity_search("Testing the index")
# print(docs[0].page_content[:1000])