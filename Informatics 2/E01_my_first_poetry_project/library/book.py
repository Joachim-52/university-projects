from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    name: str
    author: str
    pages: int

    def __repr__(self) -> str:
        return f"{self.name} by {self.author}"
