import os
import sys
import requests
from decouple import config

from langgraph.graph import StateGraph, END

from typing_extensions import TypedDict
from typing import Annotated
from pydantic import Field


try:
    OMP_API_URL = config("OMP_API_URL")
    OMP_API_TOKEN = config("OMP_API_TOKEN")
except Exception as e:
    print(f"Ero ao carregar as variavéis do .env: {e}")
    sys.exit(1)


class AgentState(TypedDict):
    path: Annotated[str, Field(description="Caminho dos arquivos")]
    archives: Annotated[list[str], Field(description="Lista de arquivos")]
    documents: Annotated[list[dict], Field(description="Documentos convertidos")]  # noqa


def get_files_for_directory(state: AgentState) -> dict:
    path = state["path"]
    archives = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(
            (
                ".pdf",
                ".docx",
            )
        )
    ]
    return {"archives": archives}


def get_file_content(file_path: str):
    with open(file_path, "rb") as file:
        file_data = file.read()

    file_name = os.path.basename(file_path)

    try:
        headers = {"Authorization": OMP_API_TOKEN}  # cabeçalho HTTP
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

    print("API:", response.status_code, response.text)
    return {"document": response_dict.get("document", "Sem dados")}


def convert_files(state: AgentState) -> dict:
    documents = []
    for file in state["archives"]:
        documents.append(get_file_content(file))
    return {"documents": documents}


def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("get_files", get_files_for_directory)
    graph.add_node("convert", convert_files)

    graph.set_entry_point("get_files")
    graph.add_edge("get_files", "convert")
    graph.add_edge("convert", END)

    return graph.compile()


if __name__ == "__main__":
    wf = create_graph()
    result = wf.invoke({"path": "data/"})
    print(result)
