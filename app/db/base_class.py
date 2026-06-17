from typing import Any
from sqlalchemy.orm import as_declarative, declared_attr
import uuid

@as_declarative()
class Base:
    id: uuid.UUID
    __name__: str

    # SQLAlchemy's declarative base generates __init__ dynamically.
    # This stub tells the type checker that column kwargs are valid.
    def __init__(self, **kwargs: Any) -> None: ...

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()