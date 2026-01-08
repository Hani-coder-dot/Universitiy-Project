class Package:
    def __init__(self, id, x, y, weight, priority):
        self.id = id #inside main would not allowed the user to enter the id
        self.x = x
        self.y = y
        self.weight = weight
        self.priority = priority

class Vehicle:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.list_of_packages = []
        self.current_load = 0#omit


vehicles_template = [
    Vehicle(id=1, capacity=60),
    Vehicle(id=2, capacity=50),
    Vehicle(id=3, capacity=40),
    Vehicle(id=4, capacity=80)
]

packages = [
    Package(id=1, x=10, y=20, weight=150, priority=2),
    Package(id=2, x=25, y=5,  weight=200, priority=1),
    Package(id=3, x=40, y=60, weight=10, priority=3),
    Package(id=4, x=15, y=35, weight=25, priority=2),
    Package(id=5, x=50, y=45, weight=18, priority=1),
    Package(id=6, x=70, y=10, weight=12, priority=2),
    Package(id=7, x=80, y=90, weight=22, priority=3),
    Package(id=8, x=60, y=15, weight=17, priority=2),
    Package(id=9, x=5,  y=5,  weight=10, priority=1),
    Package(id=10, x=30, y=70, weight=16, priority=2)
]

