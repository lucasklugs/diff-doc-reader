import pytest
from app.app import (
    valid_file,
    get_file_content,
    compare_docs_diff,
    agent_node,
    AnalysisOutput,
    AgentState,
    DiffAnalysis,
)


# ----------------------------
# TESTE: valid_file
# ----------------------------
def test_valid_file_pdf(tmp_path):
    # cria um PDF fake
    file = tmp_path / "doc.pdf"
    file.write_text("conteúdo fake")
    assert valid_file(str(file)) == str(file)


def test_valid_file_invalid_extension(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("conteúdo fake")
    with pytest.raises(Exception):
        valid_file(str(file))


# ----------------------------
# TESTE: get_file_content
# ----------------------------
def test_get_file_content_success(mocker, tmp_path):
    file = tmp_path / "arquivo.pdf"
    file.write_bytes(b"conteudo fake")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "status": "success",
        "document": "texto extraído",
    }
    mocker.patch("app.app.requests.post", return_value=mock_response)

    result = get_file_content(str(file))
    assert result["document"] == "texto extraído"


def test_get_file_content_fail(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"status": "error"}
    mocker.patch("app.app.requests.post", return_value=mock_response)

    with pytest.raises(Exception):
        get_file_content("arquivo.pdf")


# ----------------------------
# TESTE: compare_docs_diff
# ----------------------------
def test_compare_docs_diff_updates(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "status": "success",
        "differences": "<del>velho</del><ins>novo</ins>",
    }
    mocker.patch("app.app.requests.post", return_value=mock_response)

    state = {"documents": [{"document": "doc1"}, {"document": "doc2"}]}
    diffs = compare_docs_diff(state)["docs_diff"]

    assert any(d["status"] == "updated" for d in diffs)


# ----------------------------
# TESTE: agent_node
# ----------------------------
def test_agent_node_simple(mocker):
    # Criar o objeto DiffAnalysis que queremos retornar
    diff_analysis_instance = DiffAnalysis(
        diff="mudança",
        explanation="explicação",
        impact="impacto",
    )

    # Mock do modelo
    mock_model = mocker.Mock()
    mock_model.invoke.return_value = diff_analysis_instance

    # Mock do get_model() e do with_structured_output()
    mocker.patch("app.app.get_model", return_value=mock_model)
    mocker.patch.object(
        mock_model,
        "with_structured_output",
        return_value=mock_model,
    )

    # Mock do prompt
    mocker.patch("app.app.load_prompt_template", return_value="{doc_original} {doc_updated} {single_diff}")  # noqa

    state: AgentState = {
        "documents": [{"document": "doc1"}, {"document": "doc2"}],
        "docs_diff": [{"status": "added", "difference": "nova frase"}],
    }

    result = agent_node(state)

    assert len(result["analyses"]) == 1
    assert result["analyses"][0].type_diff == "added"
    assert result["analyses"][0].analysis.diff == "mudança"


# ----------------------------
# TESTE: Output final
# ----------------------------
def test_analysis_output_json():
    output = AnalysisOutput(
        doc_original="doc1",
        doc_updated="doc2",
        qtd_diff=1,
        diffs=[],
    )
    dumped = output.model_dump_json()
    assert "doc1" in dumped
    assert "doc2" in dumped
