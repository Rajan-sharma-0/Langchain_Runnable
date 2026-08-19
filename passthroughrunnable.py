from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a code generator"),
        ("human", "{topic}")
    ])

explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who explains code in simple terms"),
        ("human", "Explain the fillowing code in simplie words:\n{code}")
    ])

sequence = code_prompt | model | parser

sequence2 = RunnableParallel({
       "code": RunnablePassthrough(),
        "explaination": explain_prompt | model | parser
    }
)

chain = sequence | sequence2

response  = chain.invoke({"topic": "Please write a code of palindrome in python"})

print(response['code'])
# print(response['explaination'])
