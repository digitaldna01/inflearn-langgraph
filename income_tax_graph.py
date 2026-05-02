# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


graph_builder = StateGraph(AgentState)


# %%
def retrieve(state: AgentState):
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}


# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# %%
from langsmith import Client

client = Client()
generate_prompt = client.pull_prompt("rlm/rag-prompt")
generate_llm = ChatOpenAI(model="gpt-4o", max_completion_tokens=100)


def generate(state: AgentState):
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    # state에서 컨텍스트와 쿼리를 추출합니다
    query = state["query"]
    context = state["context"]

    # 작업 체인을 생성합니다: 먼저 프롬프트를 검색하고, 그 다음 응답을 생성합니다
    # prompt를 llm에게 주입
    rag_chain = generate_prompt | generate_llm

    # 쿼리와 컨텍스트로 체인을 호출하여 응답을 얻습니다
    response = rag_chain.invoke({"question": query, "context": context})

    # 생성된 답변을 반환합니다
    return {"answer": response.content}


# %%
from langsmith import Client
from typing import Literal

client = Client()
doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance")


def check_doc_relevance(state: AgentState) -> Literal["relevant", "irrelevant"]:
    query = state["query"]
    context = state["context"]

    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({"question": query, "documents": context})

    if response["Score"] == 1:
        # 2.3장과 다르게 `relevant`와 `irrelevant`를 반환합니다
        # node를 직접 지정하는 것보다 실제 판단 결과를 리턴하면서 해당 node의 재사용성을 높일 수 있습니다.
        return "relevant"

    return "irrelevant"


# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ["사람과 관련된 표현 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
    사전: {dictionary}
    질문 : {{query}}
""")


def rewrite(state: AgentState):
    """
    사용자의 질문을 사전을 참고하여 변경합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.
    """
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({"query": query})
    return {"query": response}


# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def check_hallucination(
    state: AgentState,
) -> Literal["hallucinated", "not hallucinated"]:
    answer = state["answer"]
    context = state["context"]
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke(
        {"student_answer": answer, "documents": context}
    )

    if response == "hallucinated":
        return "hallucinated"
    return "not hallucinated"


# %%
from langsmith import Client

client = Client()

helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness")


def check_helpfulness_grader(state: AgentState) -> str:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.

    Args:
        state (AgentState): 사용자의 질문과 생성된 답변을 포함한 에이전트의 현재 state.

    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다.
    """
    # state에서 질문과 답변을 추출합니다
    query = state["query"]
    answer = state["answer"]

    # 답변의 유용성을 평가하기 위한 체인을 생성합니다
    helpfulness_chain = helpfulness_prompt | llm

    # 질문과 답변으로 체인을 호출합니다
    response = helpfulness_chain.invoke({"question": query, "student_answer": answer})

    # 점수가 1이면 'helpful'을 반환하고, 그렇지 않으면 'unhelpful'을 반환합니다
    if response["Score"] == 1:
        return "helpful"

    return "unhelpful"


def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다.
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다

    Args:
        state (AgentState): 에이전트의 현재 state.

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """
    # 이 함수는 현재 아무 작업도 수행하지 않으며 state를 그대로 반환합니다
    return state


# %%
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("check_helpfulness", check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve", check_doc_relevance, {"relevant": "generate", "irrelevant": END}
)
graph_builder.add_conditional_edges(
    "generate",
    check_hallucination,
    {"not hallucinated": "check_helpfulness", "hallucinated": "generate"},
)

graph_builder.add_conditional_edges(
    "check_helpfulness",
    check_helpfulness_grader,
    {"helpful": END, "unhelpful": "rewrite"},
)
graph_builder.add_edge("rewrite", "retrieve")

# %%
graph = graph_builder.compile()
