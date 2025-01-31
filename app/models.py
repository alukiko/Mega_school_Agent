from pydantic import BaseModel


class UserRequest(BaseModel):
    """
    Input features validation for the ML model
    """
    query: str
    id: int
