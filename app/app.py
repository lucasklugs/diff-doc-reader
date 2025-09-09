import os
import sys
import requests
import argparse
from decouple import config
from pathlib import Path
import time

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


from typing_extensions import TypedDict
from typing import Annotated, List
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

PROMPT_PATH = Path("../promptac.md")  # chama o prompt

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
    documents: Annotated[list[dict], ...]
    prompt: Annotated[list[dict], ...]
    docs_diff: list[dict]


class DiffAnalysis(BaseModel):
    """
    Representa a análise detalhada de uma alteração detectada entre dois documentos.
    """  # noqa

    diff: str = Field(description="Diferença gerada a partir da comparação")
    explanation: str = Field(description="Explicação da mudança")
    impact: str = Field(description="Impacto dessa alteração dentro do contexto")  # noqa


class DiffContent(BaseModel):
    """
    Contém o conteúdo específico da alteração, útil para atualizações onde há valor antigo e novo.
    """  # noqa

    from_: str | None = Field(default=None, description="Alteração do documento anterior")  # noqa
    to: str | None = Field(default=None, description="Alteração do documento atualizado")  # noqa


class DiffItem(BaseModel):
    """
    Representa uma única alteração detectada, incluindo seu conteúdo e análise.
    """

    id: int = Field(description="ID da diferença")
    type_diff: str = Field(description="ID da diferença")
    content_diff: str | None | DiffContent = Field(
        default=None, description="Conteúdo da alteração ou, em caso de atualização um from to"  # noqa
    )
    analysis: DiffAnalysis = Field(description="Análise completa da diferença")


class AnalysisOutput(BaseModel):
    """
    Representa o resultado completo da análise de diferenças entre dois documentos.
    """  # noqa

    doc_original: str = Field(description="Documento Original")
    doc_updated: str = Field(description="Documento Modificado")
    qtd_diff: int = Field(description="Quantidade de modificações")
    diffs: List[DiffItem] = Field(description="modificações")


class FileConversionException(Exception):
    """Erro ao converter arquivo na API OMP."""


class DiffComparisonException(Exception):
    """Erro ao comparar documentos."""


class PromptTemplateException(Exception):
    """Erro ao pegar o prompt."""


class AgentExecutionException(Exception):
    """Erro na execução do Agente."""


class GetDocumentException(Exception):
    """Erro ao receber os documentos."""


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
            timeout=30,
        )
        response.raise_for_status()
        response_dict = response.json()
    except Exception as e:
        raise FileConversionException(f"Erro na conversão do arquivo {file_name}: {e}")  # noqa

    if response_dict.get("status") != "success":
        raise FileConversionException(
            f"API OMP falhou para {file_name}: {response_dict.get('message', 'Erro desconhecido')}"  # noqa
        )

    return {"document": response_dict.get("document", "Sem dados")}


def compare_docs_diff(state: AgentState) -> dict:
    """Compara as diferenças entre dois contratos."""
    logger.debug("Comparing contract differences...")
    if len(state.get("documents", [])) < 2:
        raise DiffComparisonException("Erro ao identificar os documentos para a comparação.")  # noqa

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
        raise DiffComparisonException(f"Erro ao chamar redlines: {e}")

    if response_dict.get("status") != "success":
        raise DiffComparisonException(
            f"Erro na comparação: {response_dict.get('message', 'Erro desconhecido')}",  # noqa
        )

    docs_diff = response_dict.get("differences")
    if not docs_diff:
        raise DiffComparisonException("Nenhuma diferença encontrada ou resposta inválida.")  # noqa

    results = []

    pattern_update = re.compile(r"<del>([^<]*)</del>\s*<ins>([^<]*)</ins>", re.DOTALL)  # noqa
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
        raise PromptTemplateException("Arquivo de prompt não encontrado.")
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as e:
        raise PromptTemplateException(f"Erro ao carregar o prompt: {e}")


def agent_node(state: AgentState) -> dict:
    try:
        model = get_model().with_structured_output(DiffAnalysis)
        prompt_template = load_prompt_template()
    except Exception as e:
        raise AgentExecutionException(f"Erro ao inicializar modelo ou prompt: {e}")  # noqa

    analyses: List[DiffItem] = []
    seen_diffs = set()

    diffs = state.get("docs_diff", [])

    for idx, diff in enumerate(diffs, start=1):
        diff_text = str(diff)
        if diff_text in seen_diffs:
            continue
        seen_diffs.add(diff_text)

        prompt = prompt_template.format(
            doc_original=state["documents"][0]["document"],
            doc_updated=state["documents"][1]["document"],
            single_diff=diff,
        )

        response: DiffAnalysis = model.invoke(prompt)
        time.sleep(6.5)

        content = diff["difference"]
        if isinstance(content, dict):
            content = DiffContent(from_=content.get("from"), to=content.get("to"))  # noqa

        analyses.append(
            DiffItem(
                id=idx,
                type_diff=diff["status"],
                content_diff=content,
                analysis=response,
            )
        )

    return {"analyses": analyses}


def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("compare", compare_docs_diff)
    graph.add_node("agent", agent_node)

    graph.set_entry_point("compare")
    graph.add_edge("compare", "agent")
    graph.add_edge("agent", END)

    return graph.compile()


def valid_file(path: str) -> str:
    """Valida se o arquivo existe e é PDF ou DOCX"""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Arquivo não encontrado: {path}")
    if not path.lower().endswith((".pdf", ".docx")):
        raise argparse.ArgumentTypeError(f"Formato inválido (somente .pdf ou .docx): {path}")  # noqa
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisador de diferenças entre documentos")  # noqa

    parser.add_argument(
        "base_file",
        type=valid_file,
        nargs="?",
        help="Informe o arquivo base para a comparação",  # noqa
    )

    parser.add_argument(
        "compare_file",
        type=valid_file,
        nargs="?",
        help="Informe o arquivo para a comparação",  # noqa
    )

    args = parser.parse_args()
    help_compare_file = parser._actions[2].help

    if args.base_file is None:
        print("É necessário pelo menos dois documentos.")
        sys.exit(1)

    if args.compare_file is None:
        print(f"{help_compare_file}")
        sys.exit(1)

    base_file = args.base_file

    compare_file = args.compare_file

    documents = [get_file_content(base_file), get_file_content(compare_file)]

    docs_diff = compare_docs_diff({"documents": documents})["docs_diff"]

    state: AgentState = {
        "documents": [get_file_content(base_file), get_file_content(compare_file)],  # noqa
        "docs_diff": compare_docs_diff({"documents": documents})["docs_diff"],
    }

    result = agent_node(state)
    analyses = result["analyses"]

    output = AnalysisOutput(
        doc_original=base_file,
        doc_updated=compare_file,
        qtd_diff=len(analyses),
        diffs=analyses,
    )

    print(output.model_dump_json(indent=2))

    folder_diff = "differences"
    os.makedirs(folder_diff, exist_ok=True)
    file_diff = os.path.join(folder_diff, "compare.json")

    with open(file_diff, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"Arquivo JSON salvo em {file_diff}")
