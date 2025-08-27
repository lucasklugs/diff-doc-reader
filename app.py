import os
import sys
import requests
from decouple import config

from langgraph.graph import StateGraph, END

from typing_extensions import TypedDict
from typing import Annotated
from pydantic import Field
import logging
import re


# Carregar variáveis de ambiente
try:
    OMP_API_URL = config("OMP_API_URL")
    OMP_API_TOKEN = config("OMP_API_TOKEN")
except Exception as e:
    print(f"Erro ao carregar as variáveis do .env: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class AgentState(TypedDict):
    path: Annotated[str, Field(description="Caminho dos arquivos")]
    archives: Annotated[list[str], Field(description="Lista de arquivos")]
    documents: Annotated[list[dict], Field(description="Documentos convertidos")]  # noqa


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


def compare_contracts_diff(state: AgentState) -> dict:
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

    contract_diff = response_dict.get("differences", "")

    results = []

    pattern_update = re.compile(r"<del>(.*?)</del>\s*<ins>(.*?)</ins>", re.DOTALL)  # noqa
    for old, new in pattern_update.findall(contract_diff):
        results.append(
            {
                "status": "updated",
                "difference": {
                    "from": old.strip(),
                    "to": new.strip(),
                },
            }
        )

    deletions = re.sub(pattern_update, "", contract_diff)
    for d in re.findall(r"<del>(.*?)</del>", deletions, re.DOTALL):
        results.append(
            {
                "status": "deleted",
                "difference": d.strip(),
            }
        )

    insertions = re.sub(pattern_update, "", contract_diff)
    for i in re.findall(r"<ins>(.*?)</ins>", insertions, re.DOTALL):
        results.append(
            {
                "status": "added",
                "difference": i.strip(),
            }
        )

    return {"contract_diff": results}


def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("get_files", get_files_for_directory)
    graph.add_node("convert", convert_files)
    graph.add_node("compare", compare_contracts_diff)

    graph.set_entry_point("get_files")
    graph.add_edge("get_files", "convert")
    graph.add_edge("convert", "compare")
    graph.add_edge("compare", END)

    return graph.compile()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python app.py data/documento1.pdf data/documento2.pdf")  # noqa
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    state: AgentState = {"path": "", "archives": [file1, file2], "documents": []}  # noqa

    state["documents"] = [get_file_content(file1), get_file_content(file2)]

    diff_result = compare_contracts_diff(state)

    print(diff_result["contract_diff"])
