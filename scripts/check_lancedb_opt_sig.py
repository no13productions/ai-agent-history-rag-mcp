
import lancedb
import inspect
import shutil
from pathlib import Path
from lancedb.pydantic import LanceModel, Vector

class TestModel(LanceModel):
    id: int
    vector: Vector(2)
    item: str

db_path = Path("./test_db_opt_sig")
if db_path.exists():
    shutil.rmtree(db_path)

try:
    db = lancedb.connect(db_path)
    table = db.create_table("test", schema=TestModel)
    
    if hasattr(table, "optimize"):
        print(f"optimize signature: {inspect.signature(table.optimize)}")
    
finally:
    if db_path.exists():
        shutil.rmtree(db_path)
