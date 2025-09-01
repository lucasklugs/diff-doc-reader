import os
import sys
import requests
from decouple import config
from pathlib import Path
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


from typing_extensions import TypedDict
from typing import Annotated, Union, List
from pydantic import Field, BaseModel
import logging
import re


# Carregar variáveis de ambiente
try:
    OMP_API_URL = config("OMP_API_URL")
    OMP_API_TOKEN = config("OMP_API_TOKEN")
    os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")
except Exception as e:
    print(f"Erro ao carregar as variáveis do .env: {e}")
    sys.exit(1)

PROMPT_PATH = Path("prompt.md")  # chama o prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        top_p=0.95,
        top_k=40,
    )


class AgentState(TypedDict):
    path: Annotated[str, Field(description="Caminho dos arquivos")]
    archives: Annotated[list[str], Field(description="Lista de arquivos")]
    documents: Annotated[list[dict], Field(description="Documentos convertidos")]  # noqa
    prompt: Annotated[list[dict], Field(description="Prompt")]


class DiffAnalysis(BaseModel):
    diff: str
    explanation: str
    impact: str


class DiffContent(BaseModel):
    from_: Union[str, None] = None  # "from" é reservado
    to: Union[str, None] = None


class DiffItem(BaseModel):
    id: int
    type_diff: str
    content_diff: Union[str, DiffContent, None]
    analysis: DiffAnalysis


class AnalysisOutput(BaseModel):
    doc_original: str
    doc_updated: str
    qtd_diff: int
    diffs: List[DiffItem]


class AgentExecutionException(Exception):
    pass


def get_files_for_directory(state: AgentState) -> dict:
    path = state["path"]
    archives = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".pdf", ".docx"))]  # noqa
    return {"archives": archives}


def get_file_content(file_path: str):
    with open(file_path, "rb") as file:
        file_data = file.read()

    file_name = os.path.basename(file_path)

    try:
        headers = {"Authorization": OMP_API_TOKEN}
        response = requests.post(
            f"{OMP_API_URL}/document-converter/0",
            files={"file": (file_name, file_data, "application/pdf")},
            headers=headers,
        )
        response_dict = response.json()
    except Exception as e:
        print(f"Erro na conversão: {e}")
        return {"document": "Erro na conversão"}

    if response_dict.get("status") != "success":
        return {"document": "Erro na API OMP"}

    return {"document": response_dict.get("document", "Sem dados")}


def convert_files(state: AgentState) -> dict:
    documents = []
    for file in state["archives"]:
        documents.append(get_file_content(file))
    return {"documents": documents}


def compare_docs_diff(state: AgentState) -> dict:
    """Compara as diferenças entre dois contratos."""
    logger.debug("Comparing contract differences...")

    url = f"{OMP_API_URL}/redlines/"
    headers = {
        "Authorization": OMP_API_TOKEN,
        "Content-Type": "application/json",
    }
    data = {
        "base_document": state["documents"][0]["document"],
        "document": state["documents"][1]["document"],
        "output_format": "markdown_none",
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        response_dict = response.json()
    except Exception as e:
        raise AgentExecutionException(f"Erro ao chamar redlines: {e}")

    if response_dict.get("status") != "success":
        raise AgentExecutionException(
            f"Erro ao comparar contratos: {response_dict.get('message', 'Erro desconhecido')}"  # noqa
        )

    docs_diff = response_dict.get("differences")

    results = []

    pattern_update = re.compile(r"<del>(.*?)</del>\s*<ins>(.*?)</ins>", re.DOTALL)  # noqa
    for old, new in pattern_update.findall(docs_diff):
        results.append(
            {
                "status": "updated",
                "difference": {
                    "from": old.strip(),
                    "to": new.strip(),
                },
            }
        )

    deletions = re.sub(pattern_update, "", docs_diff)
    for d in re.findall(r"<del>(.*?)</del>", deletions, re.DOTALL):
        results.append(
            {
                "status": "deleted",
                "difference": d.strip(),
            }
        )

    insertions = re.sub(pattern_update, "", docs_diff)
    for i in re.findall(r"<ins>(.*?)</ins>", insertions, re.DOTALL):
        results.append(
            {
                "status": "added",
                "difference": i.strip(),
            }
        )

    return {"docs_diff": results}


def load_prompt_template() -> ChatPromptTemplate:
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError("Arquivo de prompt não encontrado.")
    content = PROMPT_PATH.read_text(encoding="utf-8")
    return content


def agent_node(state: AgentState, answer: AIMessage) -> dict:
    model = get_model().with_structured_output(DiffAnalysis)
    prompt_template = load_prompt_template()

    doc1 = state["documents"][0]["document"]
    doc2 = state["documents"][1]["document"]
    diffs = state.get("docs_diff", [])

    analyses: List[DiffItem] = []
    seen_diffs = set()

    for idx, diff in enumerate(diffs, start=1):
        diff_text = str(diff)
        if diff_text in seen_diffs:
            continue
        seen_diffs.add(diff_text)

        prompt = prompt_template.format(
            doc_original=doc1,
            doc_updated=doc2,
            single_diff=diff,
        )

        response: DiffAnalysis = model.invoke(prompt)
        time.sleep(6.5)

        conteudo = diff["difference"]
        if isinstance(conteudo, dict):
            conteudo = DiffContent(from_=conteudo.get("from"), to=conteudo.get("to"))  # noqa

        analyses.append(
            DiffItem(
                id=idx,
                type_diff=diff["status"],
                content_diff=conteudo,
                analysis=response,
            )
        )

    return {"analises": analyses}


def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("get_files", get_files_for_directory)
    graph.add_node("convert", convert_files)
    graph.add_node("compare", compare_docs_diff)
    graph.add_node("agent", agent_node)

    graph.set_entry_point("get_files")
    graph.add_edge("get_files", "convert")
    graph.add_edge("convert", "compare")
    graph.add_edge("compare", "agent")
    graph.add_edge("agent", END)

    return graph.compile()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python app.py data/exemplo1.pdf data/exemplo2.pdf")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    state: AgentState = {"path": "", "archives": [file1, file2], "documents": []}  # noqa
    state["documents"] = [get_file_content(file1), get_file_content(file2)]
    state["docs_diff"] = compare_docs_diff(state)["docs_diff"]

    result = agent_node(state, answer=None)
    analyses = result["analises"]

    output = AnalysisOutput(
        doc_original=file1,
        doc_updated=file2,
        qtd_diff=len(analyses),
        diffs=analyses,
    )

    print(output.model_dump_json(indent=2))

    with open("compare.json", "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
