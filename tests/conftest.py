import shutil
from pathlib import Path

import pytest

from ragqa import config, store
from ragqa.evaluate import load_gold
from ragqa.synth.build import build_corpus

REPO = Path(__file__).resolve().parents[1]


def make_cfg(root: Path) -> dict:
    cfg = config.load(root / "ragqa.yaml", root=root)
    cfg["embedding"]["backend"] = "lsa"
    cfg["index"]["backend"] = "numpy"
    cfg["answer"]["backend"] = "extractive"
    return cfg


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Generate the SYNTHETIC corpus and ingest it once into a temporary project root."""
    root = tmp_path_factory.mktemp("project")
    shutil.copy(REPO / "ragqa.yaml", root / "ragqa.yaml")
    cfg = make_cfg(root)
    build_corpus(cfg["paths"]["corpus"], cfg["paths"]["gold"])
    store.ingest(cfg["paths"]["corpus"], cfg["paths"]["index"], cfg)
    return root


@pytest.fixture(scope="session")
def cfg(project):
    return make_cfg(project)


@pytest.fixture(scope="session")
def gold(cfg):
    return load_gold(cfg["paths"]["gold"])


@pytest.fixture(scope="session")
def index(cfg):
    return store.load(cfg["paths"]["index"], cfg)


@pytest.fixture
def fresh(project, tmp_path):
    """A writable copy of the project for tests that add, change or delete files."""
    root = tmp_path / "copy"
    shutil.copytree(project, root)
    return make_cfg(root)


def make_pdf(path: Path, title: str, body: str) -> Path:
    """A one-page PDF with a heading and a paragraph (for ingestion tests)."""
    from reportlab import rl_config
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    rl_config.invariant = 1
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, title)
    c.setFont("Helvetica", 10.5)
    c.drawString(72, 730, body)
    c.save()
    return path
