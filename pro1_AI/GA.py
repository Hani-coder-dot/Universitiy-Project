import random
import math
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#we should move always to better solution (better fitness)

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

class Chromosome:
    def __init__(self, packages, vehicles):
        self.packages = packages
        self.vehicles = vehicles
        self.genes = [None] * len(self.packages)
        self.package_index = {p.id: i for i, p in enumerate(self.packages)}
        self.fitness = 0
        self.skipped_map = {}  # Stores {vehicle_id: [skipped package IDs]}
        self.assign_randomly_by_vehicles()

    def assign_randomly_by_vehicles(self):
        # Step 1: Clear any previous state
        for v in self.vehicles:
            v.list_of_packages = []

        # Step 2: Shuffle packages for random distribution
        unassigned_packages = [p for p in self.packages]
        random.shuffle(unassigned_packages)

        # Step 3: Randomly assign each package to a vehicle
        for p in unassigned_packages:
            vehicle_id = random.randint(1, len(self.vehicles))
            vehicle = self.find_vehicle_by_id(vehicle_id)
            vehicle.list_of_packages.append((p.id, p.x, p.y))

        # Step 4: Sort each vehicle's route by package priority
        for v in self.vehicles:
            v.list_of_packages.sort(key=lambda pack: self.find_package_by_id(pack[0]).priority)

        # print("Initial Vehicle Assignments (sorted by priority):")
        # for v in self.vehicles:
        #     print(f"\nVehicle {v.id} (Capacity {v.capacity}):")
        #     for p_tuple in v.list_of_packages:
        #         p = self.find_package_by_id(p_tuple[0])
        #         print(f"  Package {p.id} at ({p.x},{p.y}) - Priority {p.priority}")

        # Step 5: Rebuild gene array based on sorted package order
        self.genes = [None] * len(self.packages)
        for v in self.vehicles:
            for pack in v.list_of_packages:
                self.genes[self.packages.index(self.find_package_by_id(pack[0]))] = v.id

    def calculate_route_distance(self, packages):
        if not packages:
            return 0
        depot = (0, 0)
        total_distance = self.encludiean_formula(depot, (packages[0].x, packages[0].y))
        for i in range(len(packages) - 1):
            point1 = (packages[i].x, packages[i].y)
            point2 = (packages[i + 1].x, packages[i + 1].y)
            total_distance += self.encludiean_formula(point1, point2)
        total_distance += self.encludiean_formula((packages[-1].x, packages[-1].y), depot)
        return total_distance

    def greedy_route(self, packages):
        if not packages:
            return []
        unvisited = packages[:]
        route = []
        current = (0, 0)
        while unvisited:
            next_pkg = min(unvisited, key=lambda p: self.encludiean_formula(current, (p.x, p.y)))
            route.append(next_pkg)
            current = (next_pkg.x, next_pkg.y)
            unvisited.remove(next_pkg)
        return route


    def find_package_by_id(self, package_id):
        for p in self.packages:
            if p.id == package_id:
                return p
        return None

    def find_vehicle_by_id(self, vehicle_id):
        for v in self.vehicles:
            if v.id == vehicle_id:
                return v
        return None

    def rebuild_vehicle_route(self):
        valid_vehicle_ids = {v.id for v in self.vehicles}

        # Fix invalid genes
        for i, v_id in enumerate(self.genes):
            if v_id not in valid_vehicle_ids:
                self.genes[i] = random.choice(list(valid_vehicle_ids))

        for v in self.vehicles:
            v.list_of_packages = []
            v.current_load = 0

        vehicle_capacity_left = {v.id: v.capacity for v in self.vehicles}
        vehicle_packages_map = {v.id: [] for v in self.vehicles}
        vehicle_skipped_map = {v.id: [] for v in self.vehicles}

        sorted_packages = sorted(self.packages, key=lambda p: p.priority)

        for p in sorted_packages:
            v_id = self.genes[self.package_index[p.id]]
            v = self.find_vehicle_by_id(v_id)

            # Try original vehicle
            if v and vehicle_capacity_left[v.id] >= p.weight:
                vehicle_packages_map[v.id].append(p)
                vehicle_capacity_left[v.id] -= p.weight
                v.current_load += p.weight
                continue

            # Try other vehicles
            reassigned = False
            for alt in self.vehicles:
                if vehicle_capacity_left[alt.id] >= p.weight:
                    vehicle_packages_map[alt.id].append(p)
                    vehicle_capacity_left[alt.id] -= p.weight
                    alt.current_load += p.weight
                    self.genes[self.package_index[p.id]] = alt.id
                    reassigned = True
                    break

            if not reassigned:
                self.genes[self.package_index[p.id]] = None
                vehicle_skipped_map[v_id].append(p)

        self.skipped_map = {
            v_id: [p.id for p in skipped_list]
            for v_id, skipped_list in vehicle_skipped_map.items()
        }

        # Route optimization
        for v in self.vehicles:
            assigned = vehicle_packages_map[v.id]
            priority_order = sorted(assigned, key=lambda p: p.priority)
            dist_priority = self.calculate_route_distance(priority_order)
            distance_order = self.greedy_route(assigned)
            dist_distance = self.calculate_route_distance(distance_order)

            chosen = distance_order if dist_distance < 0.8 * dist_priority else priority_order
            v.list_of_packages = [(p.id, p.x, p.y) for p in chosen]

        # Clean dangling genes
        for i, v_id in enumerate(self.genes):
            if v_id is None:
                continue
            v = self.find_vehicle_by_id(v_id)
            if not any(p[0] == self.packages[i].id for p in v.list_of_packages):
                self.genes[i] = None


    @staticmethod
    def encludiean_formula(point1, point2):
        x_diff = point1[0] - point2[0]
        y_diff = point1[1] - point2[1]
        return math.sqrt(x_diff**2 + y_diff**2)

    def calculate_fitness(self):
        self.rebuild_vehicle_route()
        total_distance = 0
        over_load = 0

        for v in self.vehicles:
            if v.current_load > v.capacity:
                overload_amount = v.current_load - v.capacity
                over_load += overload_amount
            if v.list_of_packages:
                depot = (0, 0)
                total_distance += self.encludiean_formula(depot, v.list_of_packages[0][1:])

                for i in range(len(v.list_of_packages) - 1):
                    point1 = v.list_of_packages[i][1:]
                    point2 = v.list_of_packages[i + 1][1:]
                    total_distance += self.encludiean_formula(point1, point2)

                total_distance += self.encludiean_formula(depot, v.list_of_packages[-1][1:])

        penalty = over_load * 1000

        # Count how many packages were skipped (unassigned)
        skipped_count = sum(1 for g in self.genes if g is None)

        # Add small penalty for skipping packages to encourage full utilization
        skip_penalty = skipped_count * 10

        self.fitness = 1 / (total_distance + penalty + skip_penalty + 1)


    def mutate(self):
        mutated_count = 0
        #index = p_id - 1
        if len(self.vehicles) == 1:
            return  # No mutation possible if only one vehicle
        random_index = random.randint(0, len(self.packages) - 1)
        current_vehicle = self.genes[random_index]
        
        new_random_vehicle = current_vehicle
        while new_random_vehicle == current_vehicle:
            new_random_vehicle = random.randint(1, len(self.vehicles))

        self.genes[random_index] = new_random_vehicle 

    def crossover(self, other):
        cut_point = random.randint(1,len(self.packages)-1)

        parent1_genes = self.genes
        parent2_genes = other.genes

        child1_genes = parent1_genes[:cut_point] + parent2_genes[cut_point:]

        child2_genes = parent2_genes[:cut_point] + parent1_genes[cut_point:]

        new_vehicles1 = [Vehicle(v.id, v.capacity) for v in self.vehicles]
        new_vehicles2 = [Vehicle(v.id, v.capacity) for v in self.vehicles]

        child1 = Chromosome(self.packages, new_vehicles1)
        child1.genes = child1_genes[:]
        #child1.rebuild_vehicle_route()
        child1.calculate_fitness()

        child2 = Chromosome(self.packages, new_vehicles2)
        child2.genes = child2_genes[:]
        #child2.rebuild_vehicle_route()
        child2.calculate_fitness()

        return child1, child2
    
    def __repr__(self):
        output = ""
        for v in self.vehicles:
            output += f"Vehicle {v.id} (Cap: {v.capacity}, Load: {v.current_load}):\n"
            for p in v.list_of_packages:
                pkg = self.find_package_by_id(p[0])
                output += f"  Package {p[0]} at ({p[1]},{p[2]}) | Priority: {pkg.priority}\n"
        output += f"Fitness: {self.fitness:.4f}\n"

        return output
    def calculate_threshold_cost(self):
    # Calculate the cost using a greedy algorithm
        total_cost = 0
        for v in self.vehicles:
            # Calculate the total distance using the greedy route
            route = self.greedy_route([self.find_package_by_id(p[0]) for p in v.list_of_packages])
            total_cost += self.calculate_route_distance(route)

        # Define the threshold as the total cost
        threshold_cost = total_cost
        print(f"Threshold cost: {threshold_cost:.2f}")
        return threshold_cost


def mutate_population(population, mutation_rate):
    num_to_mutate = int(len(population) * mutation_rate)
    selected_to_mutate = random.sample(population, num_to_mutate)
    
    for indvidual in selected_to_mutate:
        indvidual.mutate()

def create_initial_population(population_size, packages, vehicles_template):
    population = []
    for _ in range(population_size):
        vehicles = [Vehicle(v.id, v.capacity) for v in vehicles_template]
        individual = Chromosome(packages, vehicles)
        individual.calculate_fitness()
        population.append(individual)
    return population

#Tournament Selection
def select_parents(population, tournament_size=3):
    def tournament():
        compatitors = random.sample(population, tournament_size)
        best = max(compatitors, key=lambda ind: ind.fitness)
        return best
    
    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2

def create_next_generation(population, mutation_rate):
    next_population = []

    #since each two parent will make tw, so the increasing in popultion would be 2 times faster
    for _ in range(len(population)//2):
        parent1, parent2 = select_parents(population)
        child1, child2 = parent1.crossover(parent2)
        
        # for v in self.vehicles:
        #     v.list_of_packages.sort(key=lambda pack: self.find_package_by_id(pack[0]).priority) 

        next_population.append(child1)

        #happens when the population is odd number
        if len(next_population) < len (population):
            next_population.append(child2)


    mutate_population(next_population, mutation_rate)
    return next_population

def run_genetic_algorithm(packages_input, vehicles_input, generations=500, population_size=100, mutation_rate=0.05, seed=None):
    if seed is not None:
        random.seed(seed)

    print(">>> ENTERED run_genetic_algorithm", )

    packages = [Package(*p) for p in packages_input]
    vehicles_template = [Vehicle(*v) for v in vehicles_input]

    print(">>> Creating initial population...", )
    population = create_initial_population(population_size, packages, vehicles_template)
    print(">>> Initial population created", )

    generations_best = []
    checkpoints = [1, generations//5, 2*generations//5, 3*generations//5, 4*generations//5, generations]

    print(">>> Starting evolution loop", )
    for generation in range(1, generations + 1):
        print(f">>> Generation {generation} start", )
        population = create_next_generation(population, mutation_rate)
        best = max(population, key=lambda c: c.fitness)

        # #to calculate threshold cost
        # threshold = best.calculate_threshold_cost(chromosome)
        # print(f"[Generation {generation}] Threshold: {threshold:.4f}")
        
        # Print skipped packages for best chromosome only
        for v_id, skipped_ids in best.skipped_map.items():
            if skipped_ids:
                print(f"[Vehicle {v_id}] Skipped packages due to capacity: {skipped_ids}")

        print(f"[Generation {generation}, Summary]\n{best}", )

        if generation in checkpoints:
            generations_best.append((generation, best))

    #print(">>> Plotting now...", )
    #plot_selected_generations(generations_best)
    #print(">>> Done plotting.", )
    print(">>> Displaying 3 route visualizations...")
    display_chromosome_details(generations_best[-1][1], title="Final Generation")



def display_chromosome_details(chromosome, title="Route Summary"):
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.cm as cm

    root = tk.Toplevel()
    root.title(title)
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('tab10')

    skipped_capacity = []
    assigned_ids = set()

    for i, v in enumerate(chromosome.vehicles):
        if not v.list_of_packages:
            continue
        route = [(0, 0)] + [(p[1], p[2]) for p in v.list_of_packages] + [(0, 0)]
        x_vals = [pt[0] for pt in route]
        y_vals = [pt[1] for pt in route]
        ax.plot(x_vals, y_vals, marker='o', label=f"V{v.id}", color=cmap(i % 10))

        for j, (x, y) in enumerate(route):
            label = "Depot" if j == 0 or j == len(route) - 1 else f"P{v.list_of_packages[j - 1][0]}"
            ax.text(x + 1, y + 1, label, fontsize=9)

        assigned_ids.update(p[0] for p in v.list_of_packages)

    all_ids = set(p.id for p in chromosome.packages)
    skipped_all = all_ids - assigned_ids
    skipped_priority = []
    for sid in skipped_all:
        p = chromosome.find_package_by_id(sid)
        gene = chromosome.genes[chromosome.package_index[sid]]
        if gene is None:
            for v_id, skipped_ids in chromosome.skipped_map.items():
                if sid in skipped_ids:
                    skipped_capacity.append(f"P{sid} (W:{p.weight}) via V{v_id}")
                    break
        else:
            skipped_priority.append(f"P{sid} (W:{p.weight})")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Vehicle Routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()

    summary = tk.Text(root, height=10, width=90, bg='#f0f0ff')
    summary.pack()
    
    summary.insert('end', f"Total Distance: {calculate_total_distance(chromosome):.2f}\n\n")
    summary.insert('end', f"Threshold cost: {chromosome.calculate_threshold_cost():.2f}\n\n")
    for v in chromosome.vehicles:
        pkg_infos = []
        for p in v.list_of_packages:
            pkg_obj = chromosome.find_package_by_id(p[0])
            pkg_infos.append(f"P{pkg_obj.id} (W:{pkg_obj.weight})")
        summary.insert('end', f"Vehicle {v.id} (Cap: {v.capacity}, Load: {v.current_load}): {', '.join(pkg_infos) or 'No packages'}\n")

    summary.insert('end', f"\nSkipped due to capacity:\n  {' | '.join(skipped_capacity) or 'None'}\n")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()


def calculate_total_distance(chromosome):
    dist = 0
    for v in chromosome.vehicles:
        if v.list_of_packages:
            route = [(0, 0)] + [(p[1], p[2]) for p in v.list_of_packages] + [(0, 0)]
            for i in range(len(route) - 1):
                dist += chromosome.encludiean_formula(route[i], route[i + 1])
    return dist

