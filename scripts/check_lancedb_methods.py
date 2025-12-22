
import lancedb
import shutil
from pathlib import Path
from lancedb.pydantic import LanceModel, Vector

class TestModel(LanceModel):
    id: int
    vector: Vector(2)
    item: str

# Create a temporary DB
db_path = Path("./test_db_check")
if db_path.exists():
    shutil.rmtree(db_path)

try:
    db = lancedb.connect(db_path)
    table = db.create_table("test", schema=TestModel)
    
    # Check for cleanup_old_versions
    if hasattr(table, "cleanup_old_versions"):
        print("cleanup_old_versions is available")
    else:
        print("cleanup_old_versions is NOT available")
        
    if hasattr(table, "compact_files"):
        print("compact_files is available")
    else:
        print("compact_files is NOT available")
        
finally:
    if db_path.exists():
        shutil.rmtree(db_path)
