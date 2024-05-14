# Prompt Templates

**1) Basic Template 
```python
    from langchain import PromptTemplate
    
    # Define a simple prompt template as a Python string
    
    prompt_template = PromptTemplate.from_template("""
    Human: What is the capital of {place}?
    AI: The capital of {place} is {capital}
    """)
    
    prompt = prompt_template.format(place="California", capital="Sacramento")
    
    print(prompt)
```



```python
    # No Input Variable
    no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
    print(no_input_prompt.format())
    
    # One Input Variable
    template = "Tell me a {adjective} joke."
    one_input_prompt = PromptTemplate(input_variables=["adjective"], template=template)
    print(one_input_prompt.format(adjective="funny"))
    
    # Multiple Input Variables
    multiple_input_prompt = PromptTemplate(
     input_variables=["adjective", "content"],
     template="Tell me a {adjective} joke about {content}."
    )
    
    multiple_input_prompt = multiple_input_prompt.format(adjective="funny", content="chickens")
    print(multiple_input_prompt)
```

# Working with Chat Models

The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`.

```python
    from langchain_openai import ChatOpenAI

    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    
    chat = ChatOpenAI(model_name="gpt-4o")
    
    messages = [
        SystemMessage(content="You are a tough love career coach who gets to the point and pushes your mentees to be their best."),
        HumanMessage(content="How do I become an AI engineer?")
    ]
    
    response = chat.invoke(messages)
    
    print(response.content)
```

### Invoking the llm using prompt template.
```python
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-4o")
    
    prompt_template = PromptTemplate.from_template(
        template="Write a {length} story about: {content}"
    )
    
    prompt = prompt_template.format(
        length="2-sentence",
        content="Pune, India, the hometown of the legendary data scientist, Sushant Penshanwar"
    )
    
    response = llm.invoke(input=prompt)
    
    response
```


# Output parsers

- Output parsers shape the AI's text output into a more usable form, like a database entry or a JSON object.

**Main Uses:**

1. They turn a block of text into organized data.
2. They can guide the AI on how to format its responses for consistency and ease of use.


```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers.list import ListOutputParser


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o")

prompt = PromptTemplate(
    template="List 3 {things}",
    input_variables=["things"])

response = llm.invoke(input=prompt.format(things="sports that don't use balls"))
print(response.content)

# instantiate output parser
output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

print(format_instructions)
>>> Your response should be a list of comma separated values, eg: `foo, bar, baz`


prompt = PromptTemplate(
    template="List 3 {things}.\n{format_instructions}",
    input_variables=["things"],
    partial_variables={"format_instructions": format_instructions})

output = llm.predict(text=prompt.format(things="sports that don't use balls"))

print(output)

>>> hockey, swimming, athletics
```

# 🔗 **LangChain Expression Language (LCEL) Overview**:
   - **Purpose**: Simplify building complex chains from basic components.
   - **Features**: Supports streaming, parallelism, and logging.


### 🛠️ Basic Use Case: Prompt + Model + Output Parser
   - **Common Approach**: Link a prompt template with a model.
   - **Chain Mechanism**: Using the `|` symbol, like a Unix pipe, to connect components.
   - **Process Flow**: User input → Prompt Template → Model → Output Parser.

### 🧩 Understanding the Components
   - **Step-by-Step**:
     - User input is processed by the prompt template.
     - Prompt template's output goes to the model.
     - Model's output is refined by the output parser.
   - **Example Code**: `chain = prompt | model | output_parser` shows how to combine components into a single LCEL chain.


```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Write a rap haiku about {topic}")

model = ChatOpenAI(model_name="gpt-4o")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "sunny days in San Franscisco"})
```

# Retrieval : Interface with application specific data

```python
# simple retriever
from langchain.document_loaders import TextLoader
loader = TextLoader("content/golden_hymns_of_epictetus.txt")
golden_sayings = loader.load()

```

# 🔄 **Document Loaders in LangChain**:

📋 **Wide Selection**: Numerous document loaders available. Check the [documentation](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/document_loaders) for a full list.

👣 **Usage Steps**:
   1. Choose a Document Loader from LangChain.
   2. Create an instance of the Document Loader.
   3. Employ its `load()` method to convert files into LangChain documents.


### ✂️ **Understanding Text Splitters**

🔢 **Function**: Divide long texts into smaller, coherent segments.

🔗 **Goal**: Keep related text together, fitting within the model's capacity.

### 🧩 **Using `RecursiveCharacterTextSplitter`**

🔄 **Methodology**:
   - Intelligently splits texts using multiple separators.

   - Recursively adjusts if segments are too large.

   - Ensures all parts are appropriately sized.

### 🌟 **Key Aspects of Splitting**

   - Chooses optimal separators for division.

   - Continually splits large chunks.

   - Balances chunk size by characters or tokens.

   - Maintains some overlap for context.

   - Tracks chunk starting points if needed.


```python

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 50,
    length_function = len,
    add_start_index = True
)
texts = text_splitter.split_documents(golden_sayings)

print(texts[0])
print(texts[1])
```

# 🛠️ **Creating a Vector Store Retriever**

1. **Load Documents**: Utilize a document loader for initial document retrieval.

2. **Split Texts**: Break down documents into smaller sections with a text splitter.

3. **Embedding Conversion**: Apply an embedding model to transform text chunks into vectors.

4. **Vector Store Creation**: Compile these vectors into a vector store.

🔍 **Outcome**: Your vector store is now set up to search and retrieve texts by content.

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents=texts, embedding=OpenAIEmbeddings())
```

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = """ your custom prompt text. {context} {question}"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


query = "What do grief, fear, envy, and desire stem from?"


result = qa_chain.invoke({"query": query})

result["result"]
```
# 🛠️ **Using LCEL for Retrieval**

1. **Integrate Context and Question**: The prompt template includes placeholders for context and question.

2. **Preliminary Setup**
   - Set up a retriever with an in-memory store for document retrieval.

   - Runnable components can be chained or run separately.

3. **RunnableParallel for Input Preparation**

   - Use `RunnableParallel` to combine document search results and the user's question.

   - `RunnablePassthrough` passes the user's question unchanged.

4. **Workflow Steps**

   - **Step 1**: Create `RunnableParallel` with two entries: 'context' (document results) and 'question' (user's original query).

   - **Step 2**: Feed the dictionary to the prompt component, which constructs a prompt using the user's question and retrieved documents.

   - **Step 3**: Model component evaluates the prompt with OpenAI LLM

   - **Step 4**: `Output_parser` transforms response into a readable Python string.

🔄 **End-to-End Process**: From document retrieval and prompt creation to model evaluation and output parsing, the flow seamlessly integrates various components for an effective LLM-driven response.
```python
# LECL

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | QA_CHAIN_PROMPT | llm | output_parser

chain.invoke(query)
```
