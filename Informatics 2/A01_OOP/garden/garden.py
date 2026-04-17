################################################################################
# Author:      Joachim Rath
# MatNr:       51811071
# Filename:    garden.py
# Description: Garden Planner – Implementierung der Klassen
################################################################################


class PlantingError(Exception):
    pass


class Plant:
    def __init__(self, name: str, water_required: float, tools_needed: list[str]) -> None:
        self.name = name
        self.water_required = water_required
        self.tools_needed = tools_needed

    def __str__(self) -> str:
        return f"{self.name} (needs {self.water_required}L of water and these {self.tools_needed} tools.)"


class Gardener:
    def __init__(
        self,
        name: str,
        shed_tools: list[str],
        water_in_can: float,
        experience_level: str = "beginner",
    ) -> None:
        self._name = name
        self.shed_tools = shed_tools
        self.water_in_can = water_in_can
        self.experience_level = experience_level
        self._planted: list[Plant] = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value.isalpha():
            raise NameError("[ERROR] Gardener name must only contain letters!")
        self._name = value

    def water_the_plant(self, amount: float) -> None:
        if self.experience_level == "expert":
            self.water_in_can -= amount * 0.6
        else:
            self.water_in_can -= amount

    def has_all_tools(self, plant: Plant) -> bool:
        return all(tool in self.shed_tools for tool in plant.tools_needed)

    def enough_water(self, plant: Plant) -> bool:
        needed = plant.water_required * 0.6 if self.experience_level == "expert" else plant.water_required
        return self.water_in_can >= needed

    def plant(self, plant: Plant) -> None:
        if not self.has_all_tools(plant):
            raise PlantingError(f"[ERROR] Missing tools for planting {plant.name}!")
        if not self.enough_water(plant):
            raise PlantingError(f"[ERROR] Not enough water for planting {plant.name}!")
        self._planted.append(plant)
        self.water_the_plant(plant.water_required)
