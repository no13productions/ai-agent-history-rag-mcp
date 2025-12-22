
import lancedb
import inspect
import shutil
from pathlib import Path
from lancedb.pydantic import LanceModel, Vector

class TestModel(LanceModel):
    id: int
    vector: Vector(2)
    item: str

db_path = Path("./test_db_sig")
if db_path.exists():
    shutil.rmtree(db_path)

try:
    db = lancedb.connect(db_path)
    table = db.create_table("test", schema=TestModel)
    
    if hasattr(table, "cleanup_old_versions"):
        print(f"cleanup_old_versions signature: {inspect.signature(table.cleanup_old_versions)}")
        print(f"cleanup_old_versions doc: {table.cleanup_old_versions.__doc__}")
        
finally:
    if db_path.exists():
        shutil.rmtree(db_path)
