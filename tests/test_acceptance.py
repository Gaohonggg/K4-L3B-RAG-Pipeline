import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
EVALUATION = ROOT / "group_project" / "evaluation"
REPORTS = ROOT / "reports"


def visible_files(directory: Path, extensions: set[str]) -> list[Path]:
    assert directory.is_dir(), f"Missing directory: {directory.relative_to(ROOT)}"
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in extensions
    )


def test_corpus_has_required_legal_documents():
    files = visible_files(DATA / "landing" / "legal", {".pdf", ".doc", ".docx"})
    assert len(files) >= 3, "Collect at least 3 legal/policy documents"
    assert all(path.stat().st_size > 1024 for path in files)


def test_corpus_has_required_news_with_metadata():
    files = visible_files(DATA / "landing" / "news", {".json"})
    assert len(files) >= 5, "Collect at least 5 news/article JSON files"
    required = {"url", "title", "date_crawled", "content_markdown"}
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        assert required <= item.keys(), f"{path.name} is missing required metadata"
        assert all(str(item[key]).strip() for key in required)


def test_standardized_output_covers_both_source_types():
    legal = list((DATA / "standardized" / "legal").glob("*.md"))
    news = list((DATA / "standardized" / "news").glob("*.md"))
    assert len(legal) >= 3, "Standardize all required legal documents"
    assert len(news) >= 5, "Standardize all required news articles"
    assert all(len(path.read_text(encoding="utf-8").strip()) >= 200 for path in legal + news)


def test_golden_dataset_has_15_grounded_cases():
    path = EVALUATION / "golden_dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(dataset, list) and len(dataset) >= 15
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset):
        assert required <= item.keys(), f"Golden case {index} is missing required fields"
        assert all(str(item[key]).strip() for key in required)


def test_evaluation_report_is_completed():
    report = (REPORTS / "RESULT.md").read_text(encoding="utf-8")
    assert "TODO" not in report, "Complete every TODO in the evaluation report"
    lowered = report.lower()
    for heading in (
        "điểm tổng hợp",
        "so sánh a/b",
        "các trường hợp kém nhất",
        "đề xuất cải thiện",
    ):
        assert heading in lowered
