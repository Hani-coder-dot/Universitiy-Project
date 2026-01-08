import tkinter as tk
from tkinter import simpledialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PsANDVs import vehicles_template, packages
from GA import run_genetic_algorithm
from SA import DelivarybyAnnealing, convert_from_gui

BG_COLOR = '#f0f0ff'
BUTTON_BG = '#6a5acd'
BUTTON_FG = 'white'
BUTTON_ACTIVE = '#483d8b'

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Routing App")
        self.configure(bg=BG_COLOR)
        self.geometry("1000x600")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.vehicles = [(v.id, v.capacity) for v in vehicles_template]
        self.packages = [(p.id, p.x, p.y, p.weight, p.priority) for p in packages]
        self.next_vid = max(v.id for v in vehicles_template) + 1
        self.next_pid = max(p.id for p in packages) + 1

        self.algorithm_var = tk.StringVar(value="SA")

        self.frames = {}
        for F in (SetupPage, AlgorithmPage):
            frame = F(self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame('SetupPage')
        self.frames['SetupPage'].load_initial_data()

    def show_frame(self, name):
        self.frames[name].tkraise()

class SetupPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)
        self.master = master

        top = tk.Frame(self, bg=BG_COLOR)
        top.pack(fill='x', pady=5)
        tk.Button(top, text="Reset All", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.reset_all).pack(side='left', padx=10)
        tk.Button(top, text="Next >>", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE,
                  command=lambda: master.show_frame('AlgorithmPage')).pack(side='right', padx=10)

        content = tk.Frame(self, bg=BG_COLOR)
        content.pack(fill='both', expand=True, padx=10, pady=10)
        content.columnconfigure((0,1,2), weight=1)

        vframe = tk.LabelFrame(content, text="Vehicles", bg=BG_COLOR)
        vframe.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.vlist = tk.Listbox(vframe)
        self.vlist.pack(fill='both', expand=True, padx=5, pady=(5,0))
        form_v = tk.Frame(vframe, bg=BG_COLOR)
        form_v.pack(pady=5)
        tk.Label(form_v, text="Capacity:", bg=BG_COLOR).grid(row=0, column=0, sticky='e')
        self.cap_entry = tk.Entry(form_v)
        self.cap_entry.grid(row=0, column=1, padx=5)
        tk.Button(form_v, text="Add", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.add_vehicle).grid(row=1, column=0, columnspan=2, pady=5)
        btn_v = tk.Frame(form_v, bg=BG_COLOR)
        btn_v.grid(row=2, column=0, columnspan=2, pady=3)
        tk.Button(btn_v, text="Edit", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.edit_vehicle).pack(side='left', padx=(0, 10))
        tk.Button(btn_v, text="Delete", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.delete_vehicle).pack(side='left')

        dframe = tk.LabelFrame(content, text="Depot", bg=BG_COLOR)
        dframe.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self.fig = Figure(figsize=(4,4))
        self.ax = self.fig.add_subplot(111)
        self.setup_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=dframe)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        pframe = tk.LabelFrame(content, text="Packages", bg=BG_COLOR)
        pframe.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        self.plist = tk.Listbox(pframe)
        self.plist.pack(fill='both', expand=True, padx=5, pady=(5,0))
        form_p = tk.Frame(pframe, bg=BG_COLOR)
        form_p.pack(pady=5)
        labels = ["X:", "Y:", "Weight:"]
        self.p_entries = []
        for i, lbl in enumerate(labels):
            tk.Label(form_p, text=lbl, bg=BG_COLOR).grid(row=i, column=0, sticky='e')
            e = tk.Entry(form_p)
            e.grid(row=i, column=1, padx=5)
            self.p_entries.append(e)

        tk.Label(form_p, text="Prio:", bg=BG_COLOR).grid(row=3, column=0, sticky='e')
        self.priority_var = tk.IntVar(value=1)
        radio_frame = tk.Frame(form_p, bg=BG_COLOR)
        radio_frame.grid(row=3, column=1, sticky='w')
        for i in range(1, 6):
            tk.Radiobutton(radio_frame, text=str(i), variable=self.priority_var, value=i,
                           bg=BG_COLOR, selectcolor=BG_COLOR).pack(side='left')

        tk.Button(form_p, text="Add", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.add_package).grid(row=4, column=0, columnspan=2, pady=5)
        btn_p = tk.Frame(form_p, bg=BG_COLOR)
        btn_p.grid(row=5, column=0, columnspan=2)
        tk.Button(btn_p, text="Edit", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.edit_package).pack(side='left', padx=5)
        tk.Button(btn_p, text="Delete", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE, command=self.delete_package).pack(side='left', padx=5)

    def setup_axes(self):
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.grid(True)
        self.ax.scatter(0, 0, s=80, marker='D', color='red')
        self.ax.text(2, 2, 'Depot', color='red', weight='bold')
        self.fig.canvas.draw()

    def refresh_plot(self):
        self.setup_axes()
        for _, x, y, _, _ in self.master.packages:
            self.ax.scatter(x, y, s=50)
        self.fig.canvas.draw()

    def reset_all(self):
        from PsANDVs import vehicles_template, packages
        self.master.vehicles = [(v.id, v.capacity) for v in vehicles_template]
        self.master.packages = [(p.id, p.x, p.y, p.weight, p.priority) for p in packages]
        self.master.next_vid = max(v.id for v in vehicles_template) + 1
        self.master.next_pid = max(p.id for p in packages) + 1
        self.load_initial_data()

    def add_vehicle(self):
        cap = self.cap_entry.get()
        if cap.isdigit():
            vid = self.master.next_vid
            self.master.next_vid += 1
            self.master.vehicles.append((vid, int(cap)))
            self.load_initial_data()

    def delete_vehicle(self):
        idx = self.vlist.curselection()
        if idx:
            del self.master.vehicles[idx[0]]
            self.load_initial_data()

    def edit_vehicle(self):
        idx = self.vlist.curselection()
        if idx:
            i = idx[0]
            vid, old = self.master.vehicles[i]
            new = simpledialog.askinteger("Edit Vehicle", f"New capacity for V{vid}", initialvalue=old, parent=self)
            if new is not None:
                self.master.vehicles[i] = (vid, new)
                self.load_initial_data()

    def add_package(self):
        vals = [e.get() for e in self.p_entries]
        if all(v.isdigit() for v in vals):
            x, y, w = map(int, vals)
            p = self.priority_var.get()
            pid = self.master.next_pid
            self.master.next_pid += 1
            self.master.packages.append((pid, x, y, w, p))
            self.load_initial_data()

    def delete_package(self):
        idx = self.plist.curselection()
        if idx:
            del self.master.packages[idx[0]]
            self.load_initial_data()

    def edit_package(self):
        idx = self.plist.curselection()
        if idx:
            i = idx[0]
            pid, x, y, w, p = self.master.packages[i]
            new_vals = []
            for label, val in zip(["X","Y","Weight"], [x,y,w]):
                new = simpledialog.askinteger(f"Edit {label}", f"New {label}", initialvalue=val, parent=self)
                if new is None:
                    return
                new_vals.append(new)
            new_p = simpledialog.askinteger("Edit Priority", "New Priority (1-5)", initialvalue=p, minvalue=1, maxvalue=5, parent=self)
            if new_p is None:
                return
            self.master.packages[i] = (pid, *new_vals, new_p)
            self.load_initial_data()

    def load_initial_data(self):
        self.vlist.delete(0, 'end')
        for vid, cap in self.master.vehicles:
            self.vlist.insert('end', f"V{vid}: Capacity {cap}")

        self.plist.delete(0, 'end')
        for pid, x, y, w, p in self.master.packages:
            self.plist.insert('end', f"P{pid}: ({x},{y}) W:{w} Prio:{p}")

        self.refresh_plot()

class AlgorithmPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)
        tk.Label(self, text="Choose Algorithm", font=(None, 20), bg=BG_COLOR).pack(pady=20)
        tk.Radiobutton(self, text="Simulated Annealing", variable=self.master.algorithm_var, value="SA",
                       bg=BG_COLOR, selectcolor=BG_COLOR, command=self.toggle_options).pack(anchor='w', padx=60)
        tk.Radiobutton(self, text="Genetic Algorithm", variable=self.master.algorithm_var, value="GA",
                       bg=BG_COLOR, selectcolor=BG_COLOR, command=self.toggle_options).pack(anchor='w', padx=60)

        btns = tk.Frame(self, bg=BG_COLOR)
        btns.pack(pady=20)
        tk.Button(btns, text="Back", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE,
                  command=lambda: master.show_frame('SetupPage')).pack(side='left', padx=10)
        tk.Button(btns, text="Run", bg=BUTTON_BG, fg=BUTTON_FG,
                  activebackground=BUTTON_ACTIVE,
                  command=self.run_algorithm).pack(side='left', padx=10)

        self.options_frame = tk.Frame(self, bg=BG_COLOR)
        self.options_frame.pack_forget()
        self.info_label = tk.Label(self, text="", bg=BG_COLOR, font=("Courier", 10), justify="left", anchor="w")
        self.info_label.pack(pady=10)

    def toggle_options(self):
        if self.master.algorithm_var.get() == "GA":
            self.options_frame.pack(pady=10)
        else:
            self.options_frame.pack_forget()

    def run_algorithm(self):
        packages_input = list(self.master.packages)
        vehicles_input = list(self.master.vehicles)

        print("\nVehicles to be used:", vehicles_input, flush=True)
        print("Packages to be used:", packages_input, flush=True)

        if self.master.algorithm_var.get() == "GA":
            try:
                self.info_label.config(text=(
                    "Genetic Algorithm Parameters:\n"
                    f"  Generations: 100\n"
                    f"  Population Size: 100\n"
                    f"  Mutation Rate: 0.05\n"
                    f"  Seed: 42"
                ))
                print("\nStarting Genetic Algorithm...", flush=True)
                run_genetic_algorithm(
                    packages_input=packages_input,
                    vehicles_input=vehicles_input,
                    generations=500,
                    population_size=100,
                    mutation_rate=0.05,
                   # seed=42
                )
                print("\nGA finished successfully.\n", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", str(e))

        elif self.master.algorithm_var.get() == "SA":
            try:
                self.info_label.config(text=(
                    "Simulated Annealing Parameters:\n"
                    f"  Initial Temperature: 1000\n"
                    f"  Cooling Rate: 0.95\n"
                    f"  Iterations per Temp: 100"
                ))
                print("\nStarting Simulated Annealing...", flush=True)
                sa_packages, sa_vehicles = convert_from_gui(packages_input, vehicles_input)
                annealer = DelivarybyAnnealing(sa_packages, sa_vehicles)
                best_solution, best_cost = annealer.run_annealing(
                    initial_temperature=1000,
                    cooling_rate=0.95,
                    max_iterations=3,
                    #seed=42
                )
                print("\nSA finished successfully.\n", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Error", "Please select a valid algorithm.")


# This ensures the output appears in real-time, even when run in a thread.
if __name__ == '__main__':
    app = App()
    app.mainloop()
