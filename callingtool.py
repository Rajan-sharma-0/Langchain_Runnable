      ## Tool Calling ##
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print
#1 creating a tool

@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a giving text"""
    return len(text)

tools ={
    "get_text_length" : get_text_length
}
model = ChatMistralAI(model="mistral-small-2506")

# tool binding
llm_with_tool = model.bind_tools([get_text_length])

message = []
prompt = input("You: " )
query = HumanMessage(prompt)

message.append(query)

result = llm_with_tool.invoke(message)
message.append(result)
# print(message)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message= tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    # print(message)

result = llm_with_tool.invoke(message)
print(result.content)
