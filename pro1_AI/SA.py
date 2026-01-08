import random 
import math
import copy

class package:
      def __init__ (self ,id ,weight ,priority ,x ,y):
            self.id = id
            self.weight = weight
            self.priority = priority
            self.x = x
            self.y = y
      def destination(self):
            return(self.x ,self.y)

class Vehicle():
      def __init__ (self ,idVehicle ,capOfVehicle):
            self.idVehicle = idVehicle
            self.capOfVehicle = capOfVehicle
            self.packages = []

      def load(self):
            return sum(pkg.weight for pkg in self.packages )
      
      def can_add_package (self ,package):
            return self.load() + package.weight <= self.capOfVehicle
      
      def add_package(self, package):
            if self.can_add_package(package):
                  self.packages.append(package)
                  return True
            return False
      
      # --- function to convert GUI input to SA-compatible classes ---
def convert_from_gui(packages_input, vehicles_input):
      sa_packages = [package(pid, w, prio, x, y) for pid, x, y, w, prio in packages_input]
      sa_vehicles = [Vehicle(vid, cap) for vid, cap in vehicles_input]
      return sa_packages, sa_vehicles


class DelivarybyAnnealing:
      def __init__ (self ,packages, vehicles):
            self.packages = packages
            self.vehicles = vehicles
            self.packageNotPlaced = []
            
      def generate_initial_solution(self):
            packages_copy = sorted(self.packages, key=lambda pkg: pkg.priority)
            random.shuffle(packages_copy)

            for pkg in packages_copy:
                  random.shuffle(self.vehicles)
                  placed = False

                  for vehicle in self.vehicles:
                        if vehicle.add_package(pkg):
                              placed = True
                              break
                  if not placed:
                        self.packageNotPlaced.append(pkg)
            return self.vehicles
      
      @staticmethod
      def calculate_distance(x1 ,x2 ,y1 ,y2):
            return math.sqrt ((x2 - x1)**2 + (y2 - y1)**2)
      
      def total_cost(self,vehicles):
            TotalCost = 0

            for vehicle in vehicles:
                  current_location = (0,0)
                  for pkg in vehicle.packages:
                        dest = pkg.destination()
                        TotalCost += self.calculate_distance(current_location[0],dest[0],current_location[1],dest[1])
                        current_location = dest
            return TotalCost
      def generate_neighbor(self,current_solution ):
            neighbor = copy.deepcopy(current_solution)
            if len(self.vehicles) > 1: 
                  v1 ,v2 = random.sample(neighbor ,2) 

                  if  v1.packages and  v2.packages:
                        p1 = random.choice(v1.packages)
                        p2 = random.choice(v2.packages)

                        if (v1.load() - p1.weight + p2.weight <= v1.capOfVehicle and 
                        v2.load() - p2.weight + p1.weight <= v2.capOfVehicle) :
                              
                              v1.packages.remove(p1)
                              v2.packages.remove(p2)
                              v1.packages.append(p2)
                              v2.packages.append(p1)
                        else:
                              
                              random.shuffle(v1.packages)
            else:
                  vehicle = neighbor[0]  # get the single vehicle from the list
                  if len(vehicle.packages) > 1:
                        random.shuffle(vehicle.packages)
            return neighbor
      
      def accept_solution(self,currrent_cost ,new_cost, temperature):
            if new_cost < currrent_cost:
                  return True
            else:
                  probability = math.exp((new_cost - currrent_cost)/temperature)
                  return random.random() < probability
      
      def run_annealing(self, initial_temperature, cooling_rate, max_iterations, seed=None):
            if seed is not None:
                  random.seed(seed)
            current_solution = self.generate_initial_solution()
            current_cost = self.total_cost(current_solution)
            best_solution = copy.deepcopy(current_solution)
            best_cost = current_cost
            temperature = initial_temperature 
            threshold = sum(self.greedy_route_distance(v.packages) for v in best_solution)
            
            while temperature > 1:
                  iteration = 1
                  while (iteration <= max_iterations):
                        neighbor = self.generate_neighbor(current_solution)
                        neighbor_cost = self.total_cost(neighbor)
                        

                        if self.accept_solution(current_cost ,neighbor_cost ,temperature):
                              current_solution = neighbor
                              current_cost = neighbor_cost

                              if neighbor_cost < best_cost:
                                    best_solution = copy.deepcopy(neighbor)
                                    best_cost = neighbor_cost
                        iteration += 1
                  temperature *= cooling_rate
            self.vehicles = best_solution
            print("\nDelivery paths for each vehicle:")
            for v in best_solution:
                  print(f"\nVehicle {v.idVehicle}:")
                  path = "(0,0)"
                  current_location = (0, 0)
                  for pkg in v.packages:
                        dest = pkg.destination()
                        Priority = pkg.priority
                        weight = pkg.weight
                        path += f" -> (Package at {dest[0]},{dest[1]} | {weight} Kg | Priority {Priority})"
                        current_location = dest
                  print(path)
            self.display_solution(best_cost,threshold )
            return best_solution, best_cost
      def greedy_route_distance(self, packages):
            if not packages:
                  return 0
            unvisited = packages[:]
            current = (0, 0)
            total = 0
            while unvisited:
                  next_pkg = min(unvisited, key=lambda p: self.calculate_distance(current[0], p.x, current[1], p.y))
                  total += self.calculate_distance(current[0], next_pkg.x, current[1], next_pkg.y)
                  current = (next_pkg.x, next_pkg.y)
                  unvisited.remove(next_pkg)
            return total
      
      def display_solution(self, best_cost, threshold, title="Final SA Solution"):
            import tkinter as tk
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.cm as cm

            root = tk.Toplevel()
            root.title(title)

            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            cmap = cm.get_cmap('tab10')

            for i, v in enumerate(self.vehicles):
                  if not v.packages:
                        continue
                  route = [(0, 0)] + [(p.x, p.y) for p in v.packages] + [(0, 0)]
                  x_vals = [pt[0] for pt in route]
                  y_vals = [pt[1] for pt in route]

                  ax.plot(x_vals, y_vals, marker='o', label=f"V{v.idVehicle}", color=cmap(i % 10))
                  for j, (x, y) in enumerate(route):
                        label = "Depot" if j == 0 or j == len(route) - 1 else f"P{v.packages[j - 1].id} (W:{v.packages[j - 1].weight})"
                        ax.text(x + 1, y + 1, label, fontsize=9)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_title("Vehicle Routes")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.legend()

            summary = tk.Text(root, height=10, width=80, bg='#f0f0ff')
            summary.pack()
            summary.insert('end', f"Total Distance (Cost): {best_cost:.2f}\n\n")
            summary.insert('end', f"Greedy threshold cost (Cost): {threshold:.2f}\n\n")

            for v in self.vehicles:
                  summary.insert('end', f"Vehicle {v.idVehicle} (Cap: {v.capOfVehicle}, Load: {v.load()}): ")
                  pkgs = ', '.join(f"P{p.id} (W:{p.weight})" for p in v.packages)
                  summary.insert('end', pkgs or "No packages")
                  summary.insert('end', "\n")

            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.get_tk_widget().pack()
            canvas.draw()

            summary.insert('end', f"\nSkipped due to capacity:\n")
            if self.packageNotPlaced:
                  for pkg in self.packageNotPlaced:
                        summary.insert('end', f"  P{pkg.id} (W:{pkg.weight})\n")
            else:
                  summary.insert('end', "  None\n")



def generate_scalability_test_case():
    vehicles = [Vehicle(i, 100) for i in range(10)]
    packages = []
    for i in range(100):
        weight = random.randint(5, 40)
        priority = random.randint(1, 5)
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        packages.append(package(i, weight, priority, x, y))
    return packages, vehicles

def main(packages_input, vehicles_input):
      packages = [package(pid, w, prio, x, y) for pid, x, y, w, prio in packages_input]
      vehicles = [Vehicle(vid, cap) for vid, cap in vehicles_input]

      initial_temperature = 1000
      cooling_rate = 0.95
      max_iterations = 100

      annealing = DelivarybyAnnealing(packages ,vehicles)
      best_solution, best_cost = annealing.run_annealing(initial_temperature ,cooling_rate ,max_iterations)  
      threshold = sum(annealing.greedy_route_distance(v.packages) for v in best_solution)
      
      print("The best solution")   
      print(f" the minimize cost: {best_cost}") 
      print(f"Greedy threshold cost: {threshold}")
if __name__ == "__main__":
      main()
