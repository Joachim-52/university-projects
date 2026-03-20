"""
This is just a simple example of how you can use the Library class.
A main.py file is always nice to test your code and see if everything works as expected.
This can be used for quick tests when you do not want to use the test cases for now.

The imports can look different. Again, this is just an example.
The structure and the name of the modules can be different.

Make sure that you always use "from <main module>.<module> import <class>", where the main module is either
library in all your imports.
The lines 14 to 17 are only needed in this main.py file.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from library.book import Book
from library.storage import Storage


def main() -> None:
    harry_potter = Book("Harry Potter", "J.K. Rowling", 300)
    hunger_games = Book("Hunger Games", "Suzanne Collins", 400)
    
    my_book_storage = Storage()
    my_book_storage.add_book(harry_potter)
    my_book_storage.add_book(hunger_games)

    print(my_book_storage.get_books())


if __name__ == "__main__":
    main()
