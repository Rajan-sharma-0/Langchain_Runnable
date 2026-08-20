# LangChain Runnables

Small, focused Python examples for learning LangChain Expression Language (LCEL),
tool calling, and agent workflows with Mistral.

## What is included?

| File | Demonstrates |
| --- | --- |
| `sequencerunnable.py` | A prompt -> model -> parser sequence using `|` |
| `parallelrunnable.py` | Running short and detailed chains with `RunnableParallel` |
| `passthroughrunnable.py` | Passing generated code into a second chain |
| `callingtool.py` | Defining and invoking a custom tool from an LLM tool call |
| `owntool.py` | Creating a tool with the `@tool` decorator |
| `newssummarise.py` | Searching for news with Tavily and summarizing it with Mistral |
| `Agents.py` | A manual city assistant with weather and news tools plus approval prompts |
| `Agent_noHuman.py` | A LangChain agent with weather and news tools and approval middleware |

## Requirements

- Python 3.10 or newer
- A Mistral API key
- A Tavily API key for news examples
- An OpenWeatherMap API key for weather examples

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows Command Prompt
.venv\Scripts\activate.bat

python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
```

Never commit `.env` or API keys to the repository.

## Run an example

Run a script from the project directory. Most examples call the Mistral API:

```bash
python sequencerunnable.py
python parallelrunnable.py
python passthroughrunnable.py
python newssummarise.py
```

The interactive examples accept questions in the terminal. Enter `0` to quit:

```bash
python Agents.py
python Agent_noHuman.py
```

`Agents.py` asks for approval before executing each requested tool. `Agent_noHuman.py`
uses LangChain's agent middleware for the same approval workflow.

## Core LCEL ideas

Runnables share a common interface, including `.invoke()`, `.batch()`, and `.stream()`.
The pipe operator connects them from left to right:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "machine learning"})
```

`RunnableParallel` produces multiple results from one input, while
`RunnablePassthrough` forwards input without changing it. Functions decorated with
`@tool` can be bound to a chat model and executed after the model requests them.

## Notes

These scripts are learning examples rather than a production agent. Network requests
do not currently define timeouts, and API errors should be handled more defensively
before using the examples in an application.