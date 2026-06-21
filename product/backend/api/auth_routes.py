import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import get_db
from prisma.models import User
from auth import hash_password, verify_password, create_access_token, get_current_user

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/auth", tags=["auth"])


# ──── Pydantic Schemas ────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ──── Register ────
@router.post("/register")
async def register(req: RegisterRequest, db = Depends(get_db)):
    existing = await db.user.find_unique(where={"email": req.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await db.user.create(
        data={
            "email": req.email,
            "name": req.username,
            "passwordHash": hash_password(req.password),
        }
    )

    token = create_access_token(str(user.id), user.email)
    return {
        "token": token,
        "user": {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
        },
    }


# ──── Login ────
@router.post("/login")
async def login(req: LoginRequest, db = Depends(get_db)):
    user = await db.user.find_unique(where={"email": req.email})
    if not user or not verify_password(req.password, user.passwordHash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(user.id), user.email)
    return {
        "token": token,
        "user": {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
        },
    }


# ──── Get Current User ────
@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {
        "user": {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
        }
    }
