from fastapi import FastAPI

app = FastAPI(title="Text Summarizer API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
