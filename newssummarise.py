from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

tavily_search_tool = TavilySearchResults(
    max_results=5,
)

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant
    summarize the following news into clear bullet points
    {news}
    """
)

chain = prompt | model | parser

news_result = tavily_search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news": news_result})


print(result)