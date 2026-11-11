"""Agent Skill: ``rapidocr skill install`` and the bundled ``scripts/ocr.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rapidocr
from rapidocr import skill
from rapidocr.main import main as rapidocr_main

ROOT = Path(__file__).resolve().parents[2]
REPO_SKILL = ROOT / "skills" / "rapidocrlite"


def _files(folder: Path):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes().replace(b"\r\n", b"\n")
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != skill.MARKER
    }


def test_bundled_skill_layout():
    src = skill.source_dir()
    text = (src / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: rapidocrlite\ndescription: ")
    assert (src / "scripts" / "ocr.py").is_file()


def test_wheel_bundles_skill():
    """An installed wheel must carry the skill (CMake install into rapidocr/_skills)."""

    pkg = Path(rapidocr.__file__).resolve().parent
    if pkg == (ROOT / "python" / "src" / "rapidocr").resolve():
        pytest.skip("running from the source tree")
    bundled = pkg / "_skills" / "rapidocrlite"
    assert (bundled / "SKILL.md").is_file(), f"{bundled} missing from the wheel"
    assert skill.source_dir() == bundled
    if REPO_SKILL.is_dir():
        assert _files(bundled) == _files(REPO_SKILL)


def test_install_auto_detect_and_uninstall(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    done = skill.install(home=home)
    assert done == [home / ".agents/skills/rapidocrlite", home / ".claude/skills/rapidocrlite"]
    for dest in done:
        assert _files(dest) == _files(skill.source_dir())
        assert (dest / skill.MARKER).is_file()
    # Re-install (upgrade) replaces our own copy without --force.
    assert skill.install(home=home) == done
    # A foreign folder of the same name is left alone unless forced.
    foreign = home / ".cursor/skills/rapidocrlite"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: other\n---\n", encoding="utf-8")
    with pytest.raises(skill.SkillError):
        skill.install(agents=["cursor"], home=home)
    removed = skill.uninstall(home=home)
    assert sorted(removed) == sorted(done)
    assert foreign.is_dir() and not done[0].exists()


def test_install_project_and_target(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    done = skill.install(project=proj)
    assert done == [proj / ".agents/skills/rapidocrlite", proj / ".claude/skills/rapidocrlite"]
    target = tmp_path / "custom"
    assert skill.install(targets=[target], dry_run=True) == [target / "rapidocrlite"]
    assert not target.exists()


def test_cli_subcommand(tmp_path, capsys):
    target = tmp_path / "skills"
    rapidocr_main(["skill", "install", "--target", str(target)])
    assert (target / "rapidocrlite" / "SKILL.md").is_file()
    assert "installed" in capsys.readouterr().out
    rapidocr_main(["skill", "path"])
    assert capsys.readouterr().out.strip() == str(skill.source_dir())
    with pytest.raises(SystemExit):
        rapidocr_main(["skill", "install", "--agent", "no-such-agent"])
    out = subprocess.run([sys.executable, "-m", "rapidocr.skill", "uninstall", "-t", str(target)],
                         capture_output=True, text=True, check=True).stdout
    assert "removed" in out and not (target / "rapidocrlite").exists()


def test_ocr_script_json():
    det, rec, dictionary = (os.environ.get(k) for k in ("PPOCR_TEST_DET", "PPOCR_TEST_REC", "PPOCR_TEST_DICT"))
    image = ROOT / "python" / "tests" / "data" / "hello.png"
    if not all((det, rec, dictionary, os.environ.get("PPOCR_LIBRARY"))):
        pytest.skip("set PPOCR_LIBRARY and PPOCR_TEST_* to run native OCR")
    script = skill.source_dir() / "scripts" / "ocr.py"
    cmd = [sys.executable, str(script), str(image), "--model-type", "tiny", "--no-cls", "--word-box",
           "--param", f"Det.model_path={det}", "--param", f"Rec.model_path={rec}",
           "--param", f"Rec.rec_keys_path={dictionary}", "--compact"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    records = json.loads(proc.stdout)
    assert len(records) == 1
    record = records[0]
    assert record["text"] == "Hello RapidOCR 123"
    line = record["lines"][0]
    assert line["text"] == "Hello RapidOCR 123" and line["score"] > 0.9
    assert len(line["box"]) == 4 and all(len(pt) == 2 for pt in line["box"])
    assert [w["text"] for w in record["words"][0]] == ["Hello", "RapidOCR", "123"]
    text = subprocess.run(cmd[:-1] + ["--text"], capture_output=True, text=True, encoding="utf-8", check=True)
    assert text.stdout.strip() == "Hello RapidOCR 123"
    bad = subprocess.run(cmd + [str(ROOT / "no-such-image.png")], capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode == 1
    assert "error" in json.loads(bad.stdout)[1]
