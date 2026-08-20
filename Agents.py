import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from rich import print
from tavily import TavilyClient

load_dotenv()

# --- Tools ---

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"weather in {city}: {desc}, {temp} C"


taviy_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = taviy_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"no news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n {url}\n {snippet[:100]}...")

    return f"latest news in {city}:\n\n" + "\n\n".join(news_list)


# --- Model Setup ---

model = ChatMistralAI(model="mistral-small-2506")

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = model.bind_tools([get_news, get_weather])

# --- Agent Loop ---

messages = []

print("City intelligence system")
print("type 0 to quit")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "0":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            # Step 1: Ask confirmation for ALL tool calls first
            approved_tool_calls = []
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                confirm = input(f"Agent wants to call {tool_name} Approve (ys/no): ")

                if confirm.lower() in ["y", "yes"]:
                    approved_tool_calls.append((tool_call, True))
                else:
                    print(f"Permission denied for {tool_name}.")
                    approved_tool_calls.append((tool_call, False))

            # Step 2: Execute approved tools and append tool results
            for tool_call, is_approved in approved_tool_calls:
                tool_name = tool_call['name']

                if is_approved:
                    tool_result = tools[tool_name].invoke(tool_call['args'])
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']
                    ))
                else:
                    messages.append(ToolMessage(
                        content=f"User denied permission to call tool: {tool_name}",
                        tool_call_id=tool_call['id']
                    ))

            continue

        else:
            print(result.content)
            break