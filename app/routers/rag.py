from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/context/" )
async def context():
    return {"message": "context created"}
