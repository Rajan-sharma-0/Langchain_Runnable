import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from rich import print
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

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


@wrap_tool_call
def human_approvel(request, handler):
    """Ask of human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}. Approve? (y/n): ")

    if confirm.lower() != "y":
        return ToolMessage(
            content = "Tool call denied by user.",
            tool_call_id = request.tool_call['id']
        )

    return handler(request)


agent = create_agent(
    model,
    tools= [get_weather, get_news],
    system_prompt="you are a helpful city assistant",
    middleware= [human_approvel]
)

print("City Agent | type 0 to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "0":
        break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input }]},
    )

    print("Bot : ",result['messages'][-1].content)
