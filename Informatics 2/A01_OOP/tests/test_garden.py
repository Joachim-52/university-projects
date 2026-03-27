################################################################################
# Author:      Info2 Tutors
# MatNr:       -
# File:        test_garden.py
# Description: This is the testing file for the Garden Planner task.
# Comments:    You can modify this file during development, but make sure
#              to test with the orignial file in the end.
################################################################################
import pytest

try:
    from garden import Plant
except ImportError:
    pass

try:
    from garden import PlantingError
except ImportError:
    pass

try:
    from garden import Gardener
except ImportError:
    pass


def skip_if_not_implemented_oop(
    assignment_name: str, class_name: str, function_or_property_name: str | None = None
) -> pytest.mark.skipif:
    try:
        exec(f"from {assignment_name} import {class_name}")

        if function_or_property_name is not None:
            try:
                has_function_or_property = hasattr(eval(class_name), function_or_property_name)
            except AttributeError:
                has_function_or_property = False
            if not has_function_or_property:
                return pytest.mark.skipif(
                    condition=True, reason=f'"""Function {function_or_property_name} not defined"""'
                )
    except ImportError:
        return pytest.mark.skipif(condition=True, reason=f'"""Class {class_name} not implemented"""')

    return pytest.mark.skipif(condition=False, reason="")


@skip_if_not_implemented_oop("garden", "Plant")
@skip_if_not_implemented_oop("garden", "Plant", "__str__")
def test_01_check_str() -> None:
    test_plant = Plant("Potato", 3.0, ["Shovel", "Garden trowel", "Gardening fork"])
    assert (
        str(test_plant) == "Potato (needs 3.0L of water and these ['Shovel', 'Garden trowel', 'Gardening fork'] tools.)"
    )


@skip_if_not_implemented_oop("garden", "Gardener")
def test_03_gardener_init() -> None:
    test_gardener = Gardener("Peter", ["Shovel", "Watering can", "Garden trowel", "Gardening fork", "Gloves"], 10.0)
    assert test_gardener.name == "Peter"
    assert test_gardener.shed_tools == ["Shovel", "Watering can", "Garden trowel", "Gardening fork", "Gloves"]
    assert test_gardener.water_in_can == 10.0
    assert test_gardener.experience_level == "beginner"


@skip_if_not_implemented_oop("garden", "Gardener")
def test_05_name_getter() -> None:
    test_gardener = Gardener("Moritz", ["Gloves", "Watering can"], 1.2)
    assert test_gardener.name == "Moritz"


@skip_if_not_implemented_oop("garden", "Gardener")
def test_07_name_setter() -> None:
    test_gardener = Gardener("Moritzzz", ["Hoe", "Watering can", "Scissors"], 3.4)
    with pytest.raises(NameError) as e:
        test_gardener.name = "Moritz111"
    assert str(e.value) == "[ERROR] Gardener name must only contain letters!"


@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "water_the_plant")
def test_9_water_the_plant() -> None:
    test_gardener = Gardener("Ben", ["Gloves", "Watering can", "Shovel"], 5.0, "beginner")
    test_gardener.water_the_plant(4.0)
    assert test_gardener.water_in_can == 1.0


@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "water_the_plant")
def test_11_water_the_plant() -> None:
    test_gardener = Gardener("Ben", ["Gloves", "Watering can"], 10.0, "expert")
    test_gardener.water_the_plant(4.0)
    assert test_gardener.water_in_can == 7.6


@skip_if_not_implemented_oop("garden", "Plant")
@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "has_all_tools")
def test_13_has_all_tools_true() -> None:
    test_plant = Plant("Tomato", 2.4, ["Shovel", "Gloves"])
    test_gardener = Gardener("Ben", ["Shovel", "Gloves", "Watering can", "Hoe"], 5.0)
    assert test_gardener.has_all_tools(test_plant) is True


@skip_if_not_implemented_oop("garden", "Plant")
@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "enough_water")
def test_15_enough_water() -> None:
    test_plant = Plant("Tree", 10.0, ["Shovel", "Watering can"])
    test_gardener = Gardener("Anna", ["Watering can", "Gloves"], 7.0, "expert")
    assert test_gardener.enough_water(test_plant) is True


@skip_if_not_implemented_oop("garden", "Plant")
@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "plant")
def test_17_plant() -> None:
    test_plant = Plant("Tree", 10.0, ["Shovel", "Watering can"])
    test_gardener = Gardener("Anna", ["Watering can", "Gloves", "Shovel"], 7.0, "expert")
    test_gardener.plant(test_plant)
    assert test_plant in test_gardener._planted
    assert test_gardener.water_in_can == 1.0


@skip_if_not_implemented_oop("garden", "Plant")
@skip_if_not_implemented_oop("garden", "Gardener")
@skip_if_not_implemented_oop("garden", "Gardener", "plant")
def test_18_plant_tools() -> None:
    test_plant = Plant("Tree", 10.0, ["Shovel", "Watering can"])
    test_gardener = Gardener("Anna", ["Watering can", "Gloves"], 7.0, "expert")
    with pytest.raises(PlantingError) as e:
        test_gardener.plant(test_plant)
    assert str(e.value) == "[ERROR] Missing tools for planting Tree!"
