# Structured Style

width = 5
height = 3

area = width * height

print("Area:", area)

# Procedural Style

def calculate_area(width, height):
    return width * height

area = calculate_area(5, 3)
print("Area:", area)

# OO Style

class Rectangle:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


rect = Rectangle(5, 3)
print("Area:", rect.area())