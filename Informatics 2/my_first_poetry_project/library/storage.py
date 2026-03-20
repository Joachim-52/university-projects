from library.book import Book


class Storage:
    def __init__(self) -> None:
        self._books: list[Book] = []

    @property
    def books(self) -> list[Book]:
        return self._books

    def add_book(self, book):
        self.books.append(book)
        self.books.append(book)  # whoops

    def get_books(self) -> list[Book]:
        return self.books
