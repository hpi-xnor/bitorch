from pathlib import Path
from importlib import import_module


current_dir = Path(__file__).resolve().parent
for file in current_dir.iterdir():
    # grep all python files
    if file.suffix == ".py" and file.stem != "__init__":
        print(f"importing file {file.name}")
        module = import_module(f"{__name__}.{file.stem}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, type):
                globals()[attr_name] = attr
