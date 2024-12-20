from contextlib import asynccontextmanager
from fastapi import FastAPI, Response


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Performing startup tasks...")
    # startup logic here

    yield

    print("Performing shutdown tasks...")
    # shutdown logic here


app = FastAPI(lifespan=lifespan)


@app.get("/", tags=["Health Check"])
async def root() -> Response:
    return {"status": "ok", "message": "Server is Runnings!"}


if __name__ == "__main__":
    root()