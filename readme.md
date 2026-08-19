Here is a complete, beginner-friendly, and detailed `README.md` file about LangChain Runnables. You can copy and paste this directly into a `.md` file in your project folder.

---

# 🚀 A Beginner's Guide to LangChain Runnables (LCEL)

Welcome! If you are learning LangChain, you have likely encountered the term **"Runnable"** or **"LCEL"** (LangChain Expression Language).

This guide is designed for absolute beginners. It will break down what Runnables are, the different types available, and how to implement them step-by-step with clear examples.

---

## 📑 Table of Contents

1. [What is a Runnable?](https://www.google.com/search?q=%231-what-is-a-runnable)
2. [The Standard Interface (How to run them)](https://www.google.com/search?q=%232-the-standard-interface)
3. [The Magic Pipe Operator `|](https://www.google.com/search?q=%233-the-magic-pipe-operator-)`
4. [Types of Runnables & Implementations](https://www.google.com/search?q=%234-types-of-runnables--implementations)
* [RunnableSequence (Chaining)](https://www.google.com/search?q=%23a-runnablesequence-the-basic-chain)
* [RunnableParallel (Dictionaries)](https://www.google.com/search?q=%23b-runnableparallel-running-things-together)
* [RunnablePassthrough (Passing Data)](https://www.google.com/search?q=%23c-runnablepassthrough-forwarding-data)
* [RunnableLambda (Custom Functions)](https://www.google.com/search?q=%23d-runnablelambda-custom-python-functions)


5. [Summary](https://www.google.com/search?q=%235-summary)

---

## 1. What is a Runnable?

Imagine you are building a water pipeline. Water enters through a filter, goes through a pump, and comes out of a tap.

* The Filter is a step.
* The Pump is a step.
* The Tap is a step.

In LangChain, a **Runnable** is simply one of these steps. It is any object that can accept an input, do some work, and produce an output.

Almost everything in LangChain is a Runnable:

* **Prompts:** Take a dictionary of variables and output a formatted prompt.
* **LLMs / Chat Models:** Take a prompt and output a text response.
* **Output Parsers:** Take a raw text response and convert it into a cleaner format (like a string or a Python list).

When you connect these Runnables together, you create a **Chain**.

---

## 2. The Standard Interface

Because Prompts, Models, and Parsers are all "Runnables," they all share the exact same standard commands. If you know how to run one, you know how to run them all:

* `.invoke()`: Pass a single input and get a single output. *(Most common)*
* `.batch()`: Pass a list of inputs and get a list of outputs.
* `.stream()`: Get the output piece-by-piece in real-time (like ChatGPT typing effect).

---

## 3. The Magic Pipe Operator (`|`)

LangChain Expression Language (LCEL) uses the Python OR operator `|` (called the "pipe") to connect Runnables.

Think of `|` as saying: **"Take the output of the left side, and pass it as the input to the right side."**

```python
# The input goes to Prompt -> Model -> Parser -> Final Output
chain = prompt | model | parser

```

---

## 4. Types of Runnables & Implementations

Here is a detailed breakdown of the 4 most important Runnables you will use, complete with code examples.

### A. RunnableSequence (The Basic Chain)

A `RunnableSequence` is created automatically whenever you use the `|` operator. It runs steps one after the other in a straight line.

**Usage:** When you have a simple step-by-step process.

**Implementation:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize your individual Runnables
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

# 2. Create the Sequence using the | operator
chain = prompt | model | parser

# 3. Invoke the chain
response = chain.invoke({"topic": "programming"})
print(response) 
# Output: "Why do programmers prefer dark mode? Because light attracts bugs!"

```

---

### B. RunnableParallel (Running things together)

A `RunnableParallel` allows you to run multiple Runnables at the exact same time.
*Note: In LangChain, if you put Runnables inside a standard Python dictionary `{}`, it automatically converts it into a `RunnableParallel`!*

**Usage:** When you want to ask the AI two different things based on the same input, or format data before passing it to a prompt.

**Implementation:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# Let's say we want a short joke AND a detailed poem about the same topic at the same time.

joke_chain = ChatPromptTemplate.from_template("Tell a short joke about {topic}") | model | parser
poem_chain = ChatPromptTemplate.from_template("Write a 2-line poem about {topic}") | model | parser

# Create a Parallel chain using a dictionary {}
parallel_chain = {
    "joke_result": joke_chain,
    "poem_result": poem_chain
}

response = RunnableParallel(parallel_chain).invoke({"topic": "coffee"})

# The output will be a dictionary with both results!
print(response['joke_result'])
print(response['poem_result'])

```

---

### C. RunnablePassthrough (Forwarding Data)

`RunnablePassthrough` does exactly what it sounds like: it takes the input and passes it straight through without changing it.

**Usage:** This is extremely useful when you are chaining multiple steps and need to remember the *original* input for a later step.

**Implementation:**
Let's build a chain that writes code, and then explains its own code. To explain the code, the second prompt needs to see the original question AND the generated code.

```python
from langchain_core.runnables import RunnablePassthrough

code_prompt = ChatPromptTemplate.from_template("Write python code for: {task}")
explain_prompt = ChatPromptTemplate.from_template("Explain this code: \n\n{code}")

# Step 1: Generate the code
generate_code_chain = code_prompt | model | parser

# Step 2: Complex Chain using Passthrough
final_chain = {
    # 'code' key will run the generation chain
    "code": generate_code_chain,
    # 'original_task' key will just pass the original input {"task": "..."} straight through!
    "original_task": RunnablePassthrough() 
}

response = final_chain.invoke({"task": "Fibonacci sequence"})

print(response["original_task"]) # Output: {'task': 'Fibonacci sequence'}
print(response["code"])          # Output: def fibonacci(n): ...

```

---

### D. RunnableLambda (Custom Python Functions)

Sometimes, you need to do something that LangChain doesn't have a built-in tool for (like lowering the case of a string, or doing a math calculation). A `RunnableLambda` lets you turn *any* standard Python function into a Runnable so it can be used with the `|` operator.

**Usage:** When you need custom Python logic inside your chain.

**Implementation:**

```python
from langchain_core.runnables import RunnableLambda

# 1. Create a normal Python function
def count_words(text: str) -> str:
    word_count = len(text.split())
    return f"The model generated {word_count} words.\n\nOriginal Text: {text}"

# 2. Add it to your chain!
chain = prompt | model | parser | RunnableLambda(count_words)

response = chain.invoke({"topic": "space"})

# The output of the parser goes into our custom python function!
print(response)
# Output: "The model generated 15 words. Original Text: Why did the cow go to space..."

```

---

## 5. Summary

To master LangChain, just remember these core concepts:

1. **Everything is a Runnable.** (Prompts, Models, Parsers, Functions).
2. **`|` is the glue.** It passes data from left to right.
3. **`{}` runs things in parallel.**
4. **`RunnablePassthrough()` saves your data** so it doesn't get lost in the pipeline.
5. **`RunnableLambda` lets you write custom Python code** in the middle of your AI steps.

Happy Coding! 🚀