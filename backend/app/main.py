import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, documents, retrieval
from app.core.config import settings
from app.db.migrations import run_migrations
from app.db.session import Base, engine

Base.metadata.create_all(bind=engine)
run_migrations(engine)
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.mineru_output_dir, exist_ok=True)

if settings.secret_key == "change-me-in-production" and not settings.allow_insecure_dev:
    raise RuntimeError("Unsafe SECRET_KEY detected. Please set SECRET_KEY or ALLOW_INSECURE_DEV=true for local dev.")

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(retrieval.router)


@app.get("/health")
def health():
    return {"status": "ok"}
