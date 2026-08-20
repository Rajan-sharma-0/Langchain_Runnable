from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    """Grnerate a greeting message for a user"""
    return f"Hello {name}, Welcome to the AI world"

result = get_greeting.invoke({"name":"rajan"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)
############################################################################################################################################

        ## Tool Calling ##

from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

#1 creating a tool

@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a giving text"""
    return len(text)
