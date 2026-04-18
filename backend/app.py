import uvicorn
import os
import sys

# 1. Automatically handle the Python Path (Find project root from within backend/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 2. Always resolve model path relative to the project root
model_path = os.environ.get("ASR_MODEL_DIR", "models/trained_models/20k_steps")
if not os.path.isabs(model_path):
    model_path = os.path.join(root_dir, model_path)
os.environ["ASR_MODEL_DIR"] = model_path

# 3. Import the FastAPI app
try:
    # From within the backend package, we can import from .api.main or add root to sys.path
    from api.main import app
    import models
    from database import engine
    
    # Auto-create database tables
    models.Base.metadata.create_all(bind=engine)
except ImportError:
    try:
        from backend.api.main import app
        import backend.models as models
        from backend.database import engine
        
        # Auto-create database tables
        models.Base.metadata.create_all(bind=engine)
    except ImportError as e:
        print(f"Error: Could not find the ASR modules. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"--- CDAC ASR Backend (Internal Entry Point) ---")
    print(f"Project Root: {root_dir}")
    print(f"Loading Model from: {os.environ['ASR_MODEL_DIR']}")
    print(f"Server starting at http://localhost:8000")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
