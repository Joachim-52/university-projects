################################################################################
# Author:      Joachim Rath
# MatNr:       51811071
# Filename:    bus.py
# Description: Bus Management System – Implementierung aller Klassen
################################################################################

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class BusLineAlreadyExistsError(Exception):
    pass


class AccountError(Exception):
    pass


class NotEnoughTicketsAvailableError(Exception):
    pass


class BusLineDoesNotExistError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class User:
    username: str
    password: str


class BusLine:
    def __init__(
        self,
        name: str,
        cost_per_km: float,
        origin: str,
        destination: str,
        departure_time: str,
        available_tickets: int,
    ) -> None:
        self.name = name
        self.cost_per_km = cost_per_km
        self.origin = origin
        self.destination = destination
        self.departure_time = departure_time
        self.available_tickets = available_tickets
        self.booked_tickets: int = 0

    def book_tickets(self, amount: int) -> None:
        self.available_tickets -= amount
        self.booked_tickets += amount

    def __repr__(self) -> str:
        return (
            f"BusLine(name={self.name!r}, cost_per_km={self.cost_per_km!r}, "
            f"origin={self.origin!r}, destination={self.destination!r}, "
            f"departure_time={self.departure_time!r})"
        )


@dataclass
class Booking:
    user: User
    busline: BusLine
    amount: int


# ---------------------------------------------------------------------------
# BookingSystem
# ---------------------------------------------------------------------------


class BookingSystem:
    def __init__(self, city_name: str, admin_user: User) -> None:
        self.city_name = city_name
        self.admin_user = admin_user
        self._buslines: list[BusLine] = []
        self._users: list[User] = [admin_user]
        self._bookings: list[Booking] = []

    def add_busline(
        self,
        user: User,
        name: str,
        origin: str,
        destination: str,
        cost_per_km: float,
        departure_time: str,
        available_tickets: int,
    ) -> None:
        if user != self.admin_user:
            raise PermissionError("[ERROR] Only admin can add bus lines!")
        if any(bl.name == name for bl in self._buslines):
            raise BusLineAlreadyExistsError(f"Bus line '{name}' already exists.")
        self._buslines.append(BusLine(name, cost_per_km, origin, destination, departure_time, available_tickets))

    def get_available_buslines(self) -> list[BusLine]:
        return [bl for bl in self._buslines if bl.available_tickets > 0]

    def add_user(self, username: str, password: str) -> None:
        if any(u.username == username for u in self._users):
            raise AccountError("A User with this name already exists")
        self._users.append(User(username, password))

    def login_user(self, username: str, password: str) -> User:
        for user in self._users:
            if user.username == username and user.password == password:
                return user
        raise AccountError("Invalid Username or Password")

    def make_booking(self, user: User, busline: BusLine, amount: int) -> None:
        if busline.available_tickets < amount:
            raise NotEnoughTicketsAvailableError("Not enough tickets available")
        busline.book_tickets(amount)
        self._bookings.append(Booking(user, busline, amount))

    def get_bookings_for_user(self, user: User) -> list[Booking]:
        return [b for b in self._bookings if b.user == user]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CLI:
    def __init__(self, booking_system: BookingSystem) -> None:
        self._booking_system = booking_system
        self._current_user: User | None = None

    def run(self) -> None:
        while True:
            print(f"Welcome to the ticket booking system of the city of {self._booking_system.city_name}")
            print("----------------------------------------")
            print("1. Create Account")
            print("2. Log in to existing Account")
            print("3. Exit")
            print("----------------------------------------")
            choice = input("Enter your choice: ")
            if choice == "1":
                self._create_account()
            elif choice == "2":
                if self.login():
                    self._user_menu()
            elif choice == "3":
                print("Exiting...")
                return

    def _create_account(self) -> None:
        username = input("Enter user-name: ")
        password = input("Choose a password: ")
        try:
            self._booking_system.add_user(username, password)
            print("Account created successfully!")
        except AccountError as e:
            print(str(e))

    def login(self) -> bool:
        username = input("Enter your user-name: ")
        password = input("Enter your password: ")
        try:
            self._current_user = self._booking_system.login_user(username, password)
            print(f"Hello {username} :D")
            return True
        except AccountError:
            print("Invalid Username or Password")
            return False

    def _user_menu(self) -> None:
        while True:
            print("What do you want to do?")
            print("----------------------------------------")
            print("1. List available buslines")
            print("2. Book a ticket")
            print("3. Show bookings")
            print("4. Add a busline")
            print("5. Logout")
            print("----------------------------------------")
            choice = input("Enter your choice: ")
            if choice == "1":
                self.show_available_buslines()
            elif choice == "2":
                self.booking_process()
            elif choice == "3":
                self.show_bookings()
            elif choice == "4":
                self._add_busline()
            elif choice == "5":
                print("Bye bye!")
                self._current_user = None
                return

    def show_available_buslines(self) -> None:
        buslines = self._booking_system.get_available_buslines()
        if not buslines:
            print("Currently this city does not maintain any buslines :(")
            return
        print("We are currently offering these buslines:")
        print("----------------------------------------")
        for bl in buslines:
            print(f"{bl.name}: {bl.origin} -> {bl.destination}")
        print("----------------------------------------")

    def booking_process(self) -> None:
        print("What busline do you want to ride?")
        print("----------------------------------------")
        print("0. Go Back")
        buslines = self._booking_system.get_available_buslines()
        for i, bl in enumerate(buslines, 1):
            print(f"{i}. {bl.name}: {bl.origin} -> {bl.destination}")
        print("----------------------------------------")
        choice = int(input("Enter your choice: "))
        if choice == 0:
            return
        busline = buslines[choice - 1]
        amount = int(input("Enter the amount of tickets: "))
        try:
            assert self._current_user is not None
            self._booking_system.make_booking(self._current_user, busline, amount)
            print(f"Booking for {busline.name}: {busline.origin} -> {busline.destination} successful.")
        except NotEnoughTicketsAvailableError:
            print("There are not enough tickets available.")

    def show_bookings(self) -> None:
        assert self._current_user is not None
        bookings = self._booking_system.get_bookings_for_user(self._current_user)
        aggregated: dict[str, tuple[BusLine, int]] = {}
        for booking in bookings:
            key = booking.busline.name
            if key in aggregated:
                aggregated[key] = (booking.busline, aggregated[key][1] + booking.amount)
            else:
                aggregated[key] = (booking.busline, booking.amount)
        print("You have booked tickets for:")
        print("----------------------------------------")
        for name, (busline, total) in aggregated.items():
            print(f"{name}: {busline.origin} -> {busline.destination} (Amount of tickets: {total})")
        print("----------------------------------------")

    def _add_busline(self) -> None:
        assert self._current_user is not None
        name = input("Enter busline-name: ")
        origin = input("Enter origin stop: ")
        destination = input("Enter destination stop: ")
        cost_per_km = float(input("Enter busline-length: "))
        departure_time = input("Enter busline-departure-time: ")
        available_tickets = int(input("Enter available tickets: "))
        try:
            self._booking_system.add_busline(
                self._current_user, name, origin, destination, cost_per_km, departure_time, available_tickets
            )
            print(f"{name} added!")
        except BusLineAlreadyExistsError as e:
            print(str(e))
