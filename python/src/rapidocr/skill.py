"""Install the bundled ``rapidocrlite`` Agent Skill into AI agents' skill folders.

    rapidocr skill install                 # ~/.agents/skills + detected agents (Claude Code, ...)
    rapidocr skill install --agent cursor  # only the named agent(s)
    rapidocr skill install --project       # ./.agents/skills (+ ./.claude/skills if present)
    rapidocr skill install --target DIR    # DIR/rapidocrlite
    rapidocr skill uninstall               # remove copies this command installed
    rapidocr skill path                    # print the bundled skill folder

``python -m rapidocr.skill ...`` is the same command. The skill is a folder with
``SKILL.md`` and ``scripts/ocr.py`` (JSON OCR output through the public API).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SKILL_NAME = "rapidocrlite"
MARKER = ".installed-by-rapidocr"
_PACKAGE_DIR = Path(__file__).resolve().parent

# name: (global skills dir relative to home, project skills dir, detection dir relative to home or None)
# "universal" (~/.agents/skills) is read by Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode, Amp and others.
AGENTS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "universal": (".agents/skills", ".agents/skills", None),
    "claude-code": (".claude/skills", ".claude/skills", ".claude"),
    "cursor": (".cursor/skills", ".cursor/skills", None),
    "codex": (".codex/skills", ".agents/skills", None),
    "gemini-cli": (".gemini/skills", ".agents/skills", ".gemini"),
    "github-copilot": (".copilot/skills", ".agents/skills", ".copilot"),
    "opencode": (".config/opencode/skills", ".agents/skills", ".config/opencode"),
    "grok": (".grok/skills", ".grok/skills", ".grok"),
    "openclaw": (".openclaw/skills", "skills", ".openclaw"),
    "qwen-code": (".qwen/skills", ".qwen/skills", ".qwen"),
}
# Cursor and Codex read ~/.agents/skills, so they are not auto-detected (that would load the skill twice).
_AUTO = ("universal",) + tuple(name for name, (_, _, detect) in AGENTS.items() if detect)


class SkillError(RuntimeError):
    pass


def source_dir() -> Path:
    """The bundled skill folder: in the wheel, or ``skills/rapidocrlite`` of a source checkout."""

    candidates = [
        _PACKAGE_DIR / "_skills" / SKILL_NAME,
        _PACKAGE_DIR.parents[2] / "skills" / SKILL_NAME if len(_PACKAGE_DIR.parents) > 2 else None,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "SKILL.md").is_file():
            return candidate
    raise SkillError(
        "the rapidocrlite skill files are missing from this rapidocr installation; "
        "reinstall it (pip install --force-reinstall --pre rapidocrlite) or run: npx skills add RapidAI/RapidOCRLite"
    )


def _home(home: Optional[os.PathLike] = None) -> Path:
    return Path(home).expanduser() if home is not None else Path.home()


def resolve_targets(
    agents: Sequence[str] = (),
    targets: Sequence[os.PathLike] = (),
    project: Optional[os.PathLike] = None,
    home: Optional[os.PathLike] = None,
    for_uninstall: bool = False,
) -> List[Path]:
    """Skills directories (parents of ``rapidocrlite/``) selected by the options."""

    for name in agents:
        if name not in AGENTS:
            raise SkillError(f"unknown agent {name!r}; known: {', '.join(AGENTS)}")
    out: List[Path] = [Path(t).expanduser() for t in targets]
    if project is not None:
        root = Path(project).expanduser()
        names = list(agents) or ["universal"] + (["claude-code"] if (root / ".claude").is_dir() else [])
        out += [root / AGENTS[name][1] for name in names]
    elif agents:
        h = _home(home)
        out += [h / AGENTS[name][0] for name in agents]
    elif not targets:
        h = _home(home)
        if for_uninstall:
            names: Iterable[str] = AGENTS
        else:
            names = [n for n in _AUTO if AGENTS[n][2] is None or (h / AGENTS[n][2]).is_dir()]
        out += [h / AGENTS[name][0] for name in names]
    unique: List[Path] = []
    for path in out:
        if path not in unique:
            unique.append(path)
    return unique


def _is_ours(dest: Path) -> bool:
    if (dest / MARKER).is_file():
        return True
    skill_md = dest / "SKILL.md"
    if not skill_md.is_file():
        return False
    head = skill_md.read_text(encoding="utf-8", errors="replace")[:500]
    return f"name: {SKILL_NAME}" in head


def install(
    agents: Sequence[str] = (),
    targets: Sequence[os.PathLike] = (),
    project: Optional[os.PathLike] = None,
    home: Optional[os.PathLike] = None,
    force: bool = False,
    dry_run: bool = False,
) -> List[Path]:
    """Copy the skill into each selected skills directory; returns the installed folders."""

    src = source_dir()
    installed: List[Path] = []
    for parent in resolve_targets(agents, targets, project, home):
        dest = parent / SKILL_NAME
        if dest.exists() and not force and not _is_ours(dest):
            raise SkillError(f"{dest} exists and is not a rapidocrlite skill; use --force to replace it")
        if not dry_run:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (dest / MARKER).write_text(f"{_version()}\n", encoding="utf-8")
        installed.append(dest)
    return installed


def uninstall(
    agents: Sequence[str] = (),
    targets: Sequence[os.PathLike] = (),
    project: Optional[os.PathLike] = None,
    home: Optional[os.PathLike] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Remove rapidocrlite skill folders from the selected directories (never foreign folders)."""

    removed: List[Path] = []
    for parent in resolve_targets(agents, targets, project, home, for_uninstall=True):
        dest = parent / SKILL_NAME
        if dest.is_symlink():
            if _is_ours(dest):
                if not dry_run:
                    dest.unlink()
                removed.append(dest)
        elif dest.is_dir() and _is_ours(dest):
            if not dry_run:
                shutil.rmtree(dest)
            removed.append(dest)
    return removed


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapidocr skill",
        description="Install the rapidocrlite Agent Skill (SKILL.md + scripts/ocr.py) for AI coding agents.",
    )
    sub = parser.add_subparsers(dest="command")

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--agent", "-a", action="append", default=[], choices=list(AGENTS),
                       help="agent to target (repeatable); default: ~/.agents/skills plus detected agents")
        p.add_argument("--target", "-t", action="append", default=[], metavar="DIR",
                       help="skills directory to use (repeatable); the skill goes to DIR/rapidocrlite")
        p.add_argument("--project", "-p", nargs="?", const=".", default=None, metavar="DIR",
                       help="project-level install into DIR (default .): .agents/skills, plus .claude/skills if .claude exists")
        p.add_argument("--dry-run", "-n", action="store_true", help="print what would change")

    p_install = sub.add_parser("install", help="copy the skill into agent skill folders")
    selection(p_install)
    p_install.add_argument("--force", "-f", action="store_true", help="replace a different folder of the same name")
    p_uninstall = sub.add_parser("uninstall", help="remove installed copies")
    selection(p_uninstall)
    sub.add_parser("path", help="print the bundled skill folder")
    sub.add_parser("show", help="print the bundled SKILL.md")
    sub.add_parser("agents", help="list known agents and their skills folders")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "install":
            done = install(args.agent, args.target, args.project, force=args.force, dry_run=args.dry_run)
            verb = "would install" if args.dry_run else "installed"
            for path in done:
                print(f"{verb} {path}")
            if not args.dry_run:
                print("Restart the agent (or open a new session) to load the skill.")
        elif args.command == "uninstall":
            done = uninstall(args.agent, args.target, args.project, dry_run=args.dry_run)
            verb = "would remove" if args.dry_run else "removed"
            for path in done:
                print(f"{verb} {path}")
            if not done:
                print("no installed rapidocrlite skill found")
        elif args.command == "path":
            print(source_dir())
        elif args.command == "show":
            sys.stdout.write((source_dir() / "SKILL.md").read_text(encoding="utf-8"))
        elif args.command == "agents":
            home = _home()
            for name, (global_dir, project_dir, detect) in AGENTS.items():
                auto = "auto" if name in _AUTO and (detect is None or (home / detect).is_dir()) else "-"
                print(f"{name:15} {auto:5} ~/{global_dir:28} project: {project_dir}")
        else:
            parser.print_help()
            return 2
    except SkillError as exc:
        print(f"rapidocr skill: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
