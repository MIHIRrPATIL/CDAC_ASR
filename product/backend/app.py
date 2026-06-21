import uvicorn
import os
import sys

# 1. Set a default Postgres URL if none is present to satisfy Prisma validation on startup
if not os.environ.get("DATABASE_URL") or not os.environ.get("DATABASE_URL").startswith(("postgresql://", "postgres://")):
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/postgres"

# 2. Automatically handle the Python Path (Find project root and product root)
backend_dir = os.path.dirname(os.path.abspath(__file__))
product_dir = os.path.dirname(backend_dir)
project_root = os.path.dirname(product_dir)

sys.path.append(product_dir)
sys.path.append(project_root)

# 2. Always resolve model path relative to the project root if it exists locally
model_path = os.environ.get("ASR_MODEL_DIR", "MihirRPatil/nptel-asr-phoneme-v3")
if not os.path.isabs(model_path):
    local_path = os.path.join(project_root, model_path)
    if os.path.exists(local_path):
        model_path = local_path
os.environ["ASR_MODEL_DIR"] = model_path

# 3. Import the FastAPI app
try:
    from api.main import app
except ImportError:
    try:
        from backend.api.main import app
    except ImportError as e:
        print(f"Error: Could not find the ASR modules. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"--- CDAC ASR Backend (Internal Entry Point) ---")
    print(f"Project Root: {project_root}")
    print(f"Loading Model from: {os.environ['ASR_MODEL_DIR']}")
    
    # Run the server on PORT env var (defaults to 8000 for local development)
    port = int(os.environ.get("PORT", 8000))
    print(f"Server starting at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
