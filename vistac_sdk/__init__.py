# ── src/vistac_sdk/vistac_sdk/__init__.py ──────────────────────────
from pkgutil import extend_path
import importlib, pathlib, sys

# make vistac_sdk a namespace in case there are future splits
__path__ = extend_path(__path__, __name__)

# ------------------------------------------------------------------
# expose sibling  “apps/”  as sub-module  vistac_sdk.apps
_sdk_root  = pathlib.Path(__file__).resolve().parent.parent   # …/src/vistac_sdk
_apps_dir  = _sdk_root / "apps"

if _apps_dir.is_dir():
    # ensure its parent dir is import-searchable so that plain  `import apps`
    # works (this adds …/src/vistac_sdk to sys.path only once)
    _root_str = str(_sdk_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

    # import the top-level package  "apps"  and alias it as  vistac_sdk.apps
    apps_mod = importlib.import_module("apps")
    sys.modules[f"{__name__}.apps"] = apps_mod
# ------------------------------------------------------------------
