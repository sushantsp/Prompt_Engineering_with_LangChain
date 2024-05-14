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


