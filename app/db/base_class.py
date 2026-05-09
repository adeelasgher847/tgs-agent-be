from sqlalchemy.orm import as_declarative, declared_attr
import uuid

@as_declarative()
class Base:
    id: uuid.UUID
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower() 