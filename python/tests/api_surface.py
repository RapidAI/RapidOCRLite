"""Dump the public Python API surface of a rapidocr source tree as JSON (argv[1] = .../python/src)."""
import enum, inspect, json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import rapidocr  # noqa: E402
import yaml  # noqa: E402


def sig(f):
    try:
        return str(inspect.signature(f))
    except (TypeError, ValueError):
        return None


def cls_surface(c):
    out = {"bases": [b.__name__ for b in c.__mro__[1:]]}
    if issubclass(c, enum.Enum):
        out["members"] = {m.name: repr(m.value) for m in c}
        return out
    if hasattr(c, "__dataclass_fields__"):
        out["fields"] = {k: str(v.type) for k, v in c.__dataclass_fields__.items()}
    out["methods"] = {k: sig(v) for k, v in vars(c).items()
                      if (not k.startswith("_") or k in ("__init__", "__call__")) and callable(v)}
    out["init"] = sig(c.__init__)
    return out


def schema(x):
    if isinstance(x, dict):
        return {k: schema(v) for k, v in x.items()}
    return type(x).__name__


mods = {}
for name in ["rapidocr"] + [f"rapidocr.{m}" for m in ("main", "output", "utils.typings", "typings", "utils.output")]:
    try:
        mods[name] = __import__(name, fromlist=["*"])
    except ImportError:
        pass
surface = {"all": sorted(getattr(rapidocr, "__all__", [n for n in dir(rapidocr) if not n.startswith("_")]))}
for mname, m in mods.items():
    for n, v in vars(m).items():
        if n.startswith("_") or getattr(v, "__module__", "").split(".")[0] != "rapidocr":
            continue
        key = f"{v.__module__}.{n}"
        if inspect.isclass(v):
            surface[key] = cls_surface(v)
        elif inspect.isfunction(v):
            surface[key] = sig(v)
cfg = Path(rapidocr.__file__).parent / "config.yaml"
surface["config.yaml"] = schema(yaml.safe_load(cfg.read_text()))
print(json.dumps(surface, indent=1, sort_keys=True))
