# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str  # 질문
    tax_base_equation: str  # 과세표준 계산 수식
    tax_deduction: str  # 공제액
    market_ratio: str  # 공정시작가액비율
    tax_base: str  # 과세표준 계산
    answer: str  # 세율


graph_builder = StateGraph(AgentState)

# %%
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

text_path = "./documents/real_estate_tax.txt"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n"]
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

loader = TextLoader(text_path)
document_list = loader.load_and_split(text_splitter)

from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=document_list,
    embedding=embeddings,
    collection_name="real_estate_tax_collection",
    persist_directory="./real_estate_tax_collection",
)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="real_estate_tax_collection",
    persist_directory="./real_estate_tax_collection",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
query = "5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?"

# %%
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
sllm = ChatOpenAI(model="gpt-4o-mini")

client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt")

# %%

tax_base_retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 설명해주세요.",
        ),
        ("human", "{tax_base_equation_information}"),
    ]
)


tax_base_equation_chain = tax_base_equation_prompt | llm | StrOutputParser()

tax_base_chain = {
    "tax_base_equation_information": tax_base_retrieval_chain
} | tax_base_equation_chain


def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = "주택에 대한 종합부동산세 계산시 과세표율을 계산하는 방법을 수식으로 표현해서 알려주세요."
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {"tax_base_equation": tax_base_equation}


# %%

tax_deduction_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


def get_tax_deduction(state: AgentState):
    tax_deduction_question = "주택에 대한 종합부동산세 계산시 공제금액을 알려주세요"
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {"tax_deduction": tax_deduction}


# %%
from langchain_tavily import TavilySearch
from datetime import date

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tax_market_ration_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"아래 정보를 기반으로 공정시장 가액비율을 계산해주세요.\n\nContext:\n{{context}}",
        ),
        ("human", "{query}"),
    ]
)


def get_market_ratio(state: AgentState):
    query = f"오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?"
    results = tavily_search_tool.invoke(query)
    tax_market_ratio_chain = tax_market_ration_prompt | llm | StrOutputParser()
    market_ratio = tax_market_ratio_chain.invoke({"context": results, "query": query})
    return {"market_ratio": market_ratio}


# %%
initial_state = {
    "query": query,
    "tax_base_equation": "과세표준 = (주택 공시가격 합산 - 공제금액) x 공정시장가액비율",
    "tax_deduction": "주택에 대한 종합부동산세 계산 시 공제금액은 1세대 1주택자의 경우 12억원, 법인 또는 법인으로 보는 단체의 경우 6억원, 그 외의 경우 9억원입니다.",
    "market_ratio": "2026년 주택 공시가격의 공정시장가액비율은 60%입니다.",
}

# %%
from langchain_core.prompts import PromptTemplate

tax_base_calculation_prompt = PromptTemplate.from_template("""
주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 공식 : {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}
사용자 주택 공시가격 정보 : {query}""")


def calculate_tax_base(state: AgentState):
    tax_base_equation = state["tax_base_equation"]
    tax_deduction = state["tax_deduction"]
    market_ratio = state["market_ratio"]
    query = state["query"]

    tax_base_calculation_chain = tax_base_calculation_prompt | llm | StrOutputParser()
    tax_base = tax_base_calculation_chain.invoke(
        {
            "tax_base_equation": tax_base_equation,
            "tax_deduction": tax_deduction,
            "market_ratio": market_ratio,
            "query": query,
        }
    )
    return {"tax_base": tax_base}


# %%
tax_ratio_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요

종합부동산세 세율:{context}""",
        ),
        (
            "human",
            """과세표준과 사용자가 소지한 주택의 수가 이래와 같을 때 종합부동산세를 계산해주세요

과세표준:{tax_base}
주택 수:{query}""",
        ),
    ]
)


def calculate_tax_rate(state: AgentState):
    query = state["query"]
    tax_base = state["tax_base"]
    context = retriever.invoke(query)
    tax_rate_chain = tax_ratio_calculation_prompt | llm | StrOutputParser()
    tax_rate = tax_rate_chain.invoke(
        {"context": context, "tax_base": tax_base, "query": query}
    )
    print(f"tax_rate == {tax_rate}")
    return {"answer": tax_rate}


# %%
tax_base_rate = {
    "query": query,
    "tax_base": "사용자가 가진 주택 공시가격의 합산은 다음과 같습니다:\n\n5억 + 10억 + 20억 = 35억 원\n\n이 경우 공제금액은 사용자가 1세대 1주택자가 아니므로 9억원이 적용됩니다.\n\n공정시장가액비율은 2026년에 60%로 설정되어 있습니다.\n\n과세표준을 계산하기 위해 공식을 적용하면 다음과 같습니다:\n\n과세표준 = (35억 - 9억) x 60%\n                 = 26억 x 0.6\n                 = 15.6억 원\n\n즉, 사용자의 과세표준은 15.6억 원입니다. 이 과세표준을 기준으로 종합부동산세가 계산됩니다. 종합부동산세의 실제 세부 세율은 차등적용되며, 과세표준 금액 구간에 따라 다르므로, 이를 통해 납부해야 할 최종 세금을 계산할 수 있습니다. 하지만 주어진 정보로는 세율 구간을 알 수 없으므로 정확한 세금 액수를 계산할 수 없습니다.",
}


# %%
graph_builder.add_node("get_tax_base_equation", get_tax_base_equation)
graph_builder.add_node("get_tax_deduction", get_tax_deduction)
graph_builder.add_node("get_market_ratio", get_market_ratio)
graph_builder.add_node("calculate_tax_base", calculate_tax_base)
graph_builder.add_node("calculate_tax_rate", calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "get_tax_base_equation")
graph_builder.add_edge(START, "get_tax_deduction")
graph_builder.add_edge(START, "get_market_ratio")
graph_builder.add_edge("get_tax_base_equation", "calculate_tax_base")
graph_builder.add_edge("get_tax_deduction", "calculate_tax_base")
graph_builder.add_edge("get_market_ratio", "calculate_tax_base")
graph_builder.add_edge("calculate_tax_base", "calculate_tax_rate")
graph_builder.add_edge("calculate_tax_rate", END)

# %%
graph = graph_builder.compile()
