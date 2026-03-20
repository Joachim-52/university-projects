################################################################################
# Author:      Info2 Tutors
# MatNr:       -
# File:        test_library.py
# Description: This is the testing file for the Library taks.
# Comments:    You can modify this file during development, but make sure
#              to test with the orignial file in the end.
################################################################################

try:
    from library import Book
except ImportError:
    pass

try:
    from library import Storage
except ImportError:
    pass


def test_01_check_str() -> None:
    harry_potter = Book("Harry Potter", "J.K. Rowling", 300)

    assert repr(harry_potter) == "Harry Potter written by J.K. Rowling"


def test_02_nutrition_value() -> None:
    harry_potter = Book("Harry Potter", "J.K. Rowling", 300)
    hunger_games = Book("Hunger Games", "Suzanne Collins", 400)

    my_book_storage = Storage()
    my_book_storage.add_book(harry_potter)
    my_book_storage.add_book(hunger_games)

    assert my_book_storage.get_books() == [harry_potter, hunger_games]
