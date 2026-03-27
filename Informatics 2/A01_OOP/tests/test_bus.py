################################################################################
# Author:      Info2 Tutors
# MatNr:       -
# File:        test_bus.py
# Description: This is the testing file for the Bus System task.
# Comments:    You can modify this file during development, but make sure
#              to test with the original file in the end.
################################################################################
import sys
from pathlib import Path
from typing import Any

import pytest

TEST_DATA_DIR = Path(__file__).parent / "TEST_DATA"

try:
    from bus import AccountError
except ImportError:
    pass

try:
    from bus import Booking
except ImportError:
    pass

try:
    from bus import BookingSystem
except ImportError:
    pass

try:
    from bus import BusLine
except ImportError:
    pass

try:
    from bus import BusLineAlreadyExistsError
except ImportError:
    pass

try:
    from bus import CLI
except ImportError:
    pass

try:
    from bus import NotEnoughTicketsAvailableError
except ImportError:
    pass

try:
    from bus import User
except ImportError:
    pass


def skip_if_not_implemented_oop(
    assignment_name: str, class_name: str, function_or_property_name: str | None = None
) -> Any:
    try:
        exec(f"from {assignment_name} import {class_name}")

        if function_or_property_name is not None:
            try:
                has_function_or_property = hasattr(eval(class_name), function_or_property_name)
            except AttributeError:
                has_function_or_property = False
            if not has_function_or_property:
                return pytest.mark.skipif(
                    condition=True,
                    reason=f'"""Function {function_or_property_name} not defined"""',
                )
    except ImportError:
        return pytest.mark.skipif(
            condition=True,
            reason=f'"""Class {class_name} not implemented"""',
        )

    return pytest.mark.skipif(condition=False, reason="")


try:

    @pytest.fixture
    def simple_booking_system() -> BookingSystem:
        admin_user = User("admin_user", "admin123")
        booking_system = BookingSystem("Graz", admin_user)
        return booking_system

except ImportError:
    pass
except NameError:
    pass


try:

    @pytest.fixture
    def booking_system() -> BookingSystem:
        admin_user = User("admin_user", "admin123")
        booking_system = BookingSystem("Graz", admin_user)
        booking_system.add_busline(admin_user, "Line 1", "Music Hall", "Harbour", 3, "15:30", 100)
        booking_system.add_busline(admin_user, "Line 2", "Main Station", "Cemetary", 5, "09:45", 150)
        booking_system.add_busline(admin_user, "Line 3", "City Center", "Main Square", 2.25, "08:12", 100)
        booking_system.add_busline(admin_user, "Line 4", "University", "Shopping District", 7, "12:20", 200)
        return booking_system

except ImportError:
    pass
except NameError:
    pass


try:

    @pytest.fixture
    def advanced_booking_system() -> BookingSystem:
        admin_user = User("admin", "admin")
        booking_system = BookingSystem("Graz", admin_user)
        booking_system.add_user("user_account", "***")
        booking_system.add_busline(admin_user, "Line 1", "Music Hall", "Harbour", 3, "15:30", 100)
        booking_system.add_busline(admin_user, "Line 2", "Main Station", "Cemetary", 5, "09:45", 150)
        booking_system.add_busline(admin_user, "Line 3", "City Center", "Main Square", 2.25, "08:12", 100)
        booking_system.add_busline(admin_user, "Line 4", "University", "Shopping District", 7, "12:20", 200)
        return booking_system

except ImportError:
    pass
except NameError:
    pass


@skip_if_not_implemented_oop("bus", "User")
def test_bus_01_check_user() -> None:
    user = User("Peter", "mysupersecurepassword")
    assert user.username == "Peter"
    assert user.password == "mysupersecurepassword"


@skip_if_not_implemented_oop("bus", "Booking")
def test_bus_02_check_booking() -> None:
    user = User("Peter", "secret")
    busline = BusLine("Line 7", 12.5, "Station A", "Station B", "14:30", 40)
    booking = Booking(user, busline, 3)

    assert booking.user == user
    assert booking.busline == busline
    assert booking.amount == 3


@skip_if_not_implemented_oop("bus", "BusLine")
@skip_if_not_implemented_oop("bus", "BusLine", "book_tickets")
def test_bus_04_busline_book_tickets() -> None:
    busline = BusLine("Line 2", 5, "Main Station", "Cemetary", "09:45", 150)
    busline.book_tickets(10)
    assert busline.available_tickets == 140
    assert busline.booked_tickets == 10


@skip_if_not_implemented_oop("bus", "BookingSystem")
def test_bus_06_booking_system_init() -> None:
    admin_user = User("admin", "admin")
    booking_system = BookingSystem("Graz", admin_user)
    assert booking_system.city_name == "Graz"


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "get_available_buslines")
@skip_if_not_implemented_oop("bus", "BookingSystem", "add_busline")
def test_bus_07_add_busline_and_get_available_buslines(simple_booking_system) -> None:
    assert len(simple_booking_system.get_available_buslines()) == 0
    simple_booking_system.add_busline(
        User("admin_user", "admin123"),
        "Line 1",
        "Music Hall",
        "Harbour",
        3,
        "15:30",
        100,
    )
    assert len(simple_booking_system.get_available_buslines()) == 1
    assert simple_booking_system.get_available_buslines()[0].name == "Line 1"


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "add_busline")
def test_bus_09_add_busline_existing_line_fail(booking_system) -> None:
    with pytest.raises(BusLineAlreadyExistsError):
        booking_system.add_busline(
            User("admin_user", "admin123"),
            "Line 1",
            "Music Hall",
            "Harbour",
            3,
            "15:30",
            100,
        )


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "add_user")
@skip_if_not_implemented_oop("bus", "BookingSystem", "login_user")
def test_bus_11_add_user_and_login_with_wrong_password(simple_booking_system) -> None:
    simple_booking_system.add_user("Maxim", "123456")
    with pytest.raises(AccountError, match="Invalid Username or Password"):
        simple_booking_system.login_user("Maxim", "Maxim")


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "add_user")
def test_bus_13_add_user_with_existing_user(booking_system) -> None:
    with pytest.raises(AccountError, match="A User with this name already exists"):
        booking_system.add_user("admin_user", "hello")


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "get_available_buslines")
def test_bus_15_get_available_buslines(booking_system) -> None:
    available_buslines = booking_system.get_available_buslines()
    assert len(available_buslines) == 4

    expected_available_buslines = [
        BusLine("Line 1", 3, "Music Hall", "Harbour", "15:30", 100),
        BusLine("Line 2", 5, "Main Station", "Cemetary", "09:45", 150),
        BusLine("Line 3", 2.25, "City Center", "Main Square", "08:12", 100),
        BusLine("Line 4", 7, "University", "Shopping District", "12:20", 200),
    ]

    available_buslines_repr = sorted(repr(busline) for busline in available_buslines)
    expected_available_buslines_repr = sorted(repr(busline) for busline in expected_available_buslines)

    assert available_buslines_repr == expected_available_buslines_repr


@skip_if_not_implemented_oop("bus", "BookingSystem")
@skip_if_not_implemented_oop("bus", "BookingSystem", "make_booking")
def test_bus_19_make_booking_tickets_fails(booking_system) -> None:
    existing_busline = BusLine("Line 1", 3, "Music Hall", "Harbour", "15:30", 100)

    booking_system.make_booking(User("admin_user", "admin123"), existing_busline, 10)
    booking_system.make_booking(User("admin_user", "admin123"), existing_busline, 20)
    booking_system.make_booking(User("admin_user", "admin123"), existing_busline, 25)

    with pytest.raises(NotEnoughTicketsAvailableError, match="Not enough tickets available"):
        booking_system.make_booking(User("admin_user", "admin123"), existing_busline, 50)


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "login")
def test_bus_20_cli_login(advanced_booking_system, capsys) -> None:
    cli = CLI(advanced_booking_system)
    path = Path(TEST_DATA_DIR / "preprogrammed_inputs_19.txt")
    with path.open() as sys.stdin:
        return_value = cli.login()
        your_output = capsys.readouterr().out
        expected_output = "Hello user_account :D\n"
        assert return_value is True
        assert your_output.endswith(expected_output)


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "login")
def test_bus_21_cli_login_fails(advanced_booking_system, capsys) -> None:
    cli = CLI(advanced_booking_system)
    path = Path(TEST_DATA_DIR / "preprogrammed_inputs_20.txt")
    with path.open() as sys.stdin:
        cli.login()
        your_output = capsys.readouterr().out
        expected_output = "Invalid Username or Password\n"
        assert your_output.endswith(expected_output)


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "booking_process")
def test_bus_23_cli_booking_process_go_back(advanced_booking_system, capsys) -> None:
    cli = CLI(advanced_booking_system)
    path = Path(TEST_DATA_DIR / "preprogrammed_inputs_22.txt")
    with path.open() as sys.stdin:
        return_value = cli.booking_process()
        your_output = capsys.readouterr().out
        expected_output = (
            "What busline do you want to ride?\n"
            "----------------------------------------\n"
            "0. Go Back\n"
            "1. Line 1: Music Hall -> Harbour\n"
            "2. Line 2: Main Station -> Cemetary\n"
            "3. Line 3: City Center -> Main Square\n"
            "4. Line 4: University -> Shopping District\n"
            "----------------------------------------\n"
            "Enter your choice: "
        )
        assert return_value is None
        assert your_output == expected_output


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "login")
@skip_if_not_implemented_oop("bus", "CLI", "booking_process")
def test_bus_25_cli_booking_process_fails(advanced_booking_system, capsys) -> None:
    cli = CLI(advanced_booking_system)
    path = Path(TEST_DATA_DIR / "preprogrammed_inputs_24.txt")
    with path.open() as sys.stdin:
        cli.login()
        return_value = cli.booking_process()
        your_output = capsys.readouterr().out
        expected_output = (
            "Enter your user-name: Enter your password: Hello admin :D\n"
            "What busline do you want to ride?\n"
            "----------------------------------------\n"
            "0. Go Back\n"
            "1. Line 1: Music Hall -> Harbour\n"
            "2. Line 2: Main Station -> Cemetary\n"
            "3. Line 3: City Center -> Main Square\n"
            "4. Line 4: University -> Shopping District\n"
            "----------------------------------------\n"
            "Enter your choice: Enter the amount of tickets: "
            "There are not enough tickets available.\n"
        )
        assert return_value is None
        assert your_output == expected_output


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "show_available_buslines")
def test_bus_29_show_available_buslines_empty(simple_booking_system, capsys) -> None:
    cli = CLI(simple_booking_system)
    cli.show_available_buslines()
    your_output = capsys.readouterr().out
    expected_output = "Currently this city does not maintain any buslines :(\n"
    assert your_output == expected_output


@skip_if_not_implemented_oop("bus", "CLI")
@skip_if_not_implemented_oop("bus", "CLI", "run")
def test_bus_31_run_advanced_test(advanced_booking_system, capsys) -> None:
    cli = CLI(advanced_booking_system)
    path_output = Path(TEST_DATA_DIR / "expected_output_30.txt")
    with path_output.open() as output_file_content:
        expected_output = output_file_content.read()
    path_input = Path(TEST_DATA_DIR / "preprogrammed_inputs_30.txt")
    with path_input.open() as sys.stdin:
        return_value = cli.run()
        your_output = capsys.readouterr().out
        assert return_value is None
        assert your_output == expected_output
