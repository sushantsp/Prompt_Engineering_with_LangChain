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

