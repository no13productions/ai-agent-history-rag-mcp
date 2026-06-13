import inspect
import shutil
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector


class TestModel(LanceModel):
    id: int
    vector: Vector(2)
    item: str


db_path = Path("./test_db_opt")
if db_path.exists():
    shutil.rmtree(db_path)

try:
    db = lancedb.connect(db_path)
    table = db.create_table("test", schema=TestModel)

    if hasattr(table, "optimize"):
        print(f"optimize signature: {inspect.signature(table.optimize)}")
        print(f"optimize doc: {table.optimize.__doc__}")
    else:
        print("optimize method NOT found on table")

finally:
    if db_path.exists():
        shutil.rmtree(db_path)
