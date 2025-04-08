""" Numpy matrix editor

Note that on Ubuntu tkinter isnt installed by default and you will get a
"module not found" error. You can install tkinter with:

$ sudo apt update
$ sudo apt install python3-tk

after that importing tkinter should work correctly
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sys
import os
import time # Import time for timestamp in status

# --- Configuration ---
DEFAULT_ROWS = 10
DEFAULT_COLS = 10
CELL_SIZE = 25
VALUE_COLORS = {
    0: "#FFFFFF",  # White
    1: "#ADD8E6",  # Light Blue
    2: "#90EE90",  # Light Green
    3: "#FFB6C1",  # Light Pink
}
# --- End Configuration ---

class MatrixEditorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Matrix Editor (0-3) - Drag Paint & Fill") # Updated title

        # --- Data ---
        self.rows = tk.IntVar(value=DEFAULT_ROWS)
        self.cols = tk.IntVar(value=DEFAULT_COLS)
        self.matrix_data = np.zeros((DEFAULT_ROWS, DEFAULT_COLS), dtype=int)
        self.selected_value = tk.IntVar(value=0)
        self.cell_buttons = []
        self.is_dragging = False

        # --- UI Frames ---
        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # --- Scrollable Grid ---
        self.canvas = tk.Canvas(master)
        self.vsb = ttk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(master, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.grid_frame = ttk.Frame(self.canvas) # Frame to hold the grid of buttons
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", self.on_frame_configure)
        self.grid_frame.bind("<ButtonRelease-1>", self.handle_button_release)
        self.grid_frame.bind("<ButtonRelease-3>", self.handle_right_button_release)
        self.master.bind("<ButtonRelease-1>", self.handle_button_release, add='+')
    
        # --- Bind mouse wheel events to the canvas ---
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # For vertical scrolling
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel) # For horizontal scrolling

        # --- End Scrollable Grid ---

        # --- Control Widgets ---
        control_row1 = ttk.Frame(self.control_frame)
        control_row1.pack(side=tk.TOP, fill=tk.X)
        control_row2 = ttk.Frame(self.control_frame)
        control_row2.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Row 1: Size, Resize, Save, Load
        ttk.Label(control_row1, text="Rows:").pack(side=tk.LEFT, padx=5)
        self.rows_entry = ttk.Entry(control_row1, textvariable=self.rows, width=5)
        self.rows_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_row1, text="Cols:").pack(side=tk.LEFT, padx=5)
        self.cols_entry = ttk.Entry(control_row1, textvariable=self.cols, width=5)
        self.cols_entry.pack(side=tk.LEFT, padx=5)
        self.resize_button = ttk.Button(control_row1, text="Create/Resize Grid", command=self.create_or_resize_grid)
        self.resize_button.pack(side=tk.LEFT, padx=10)
        self.save_button = ttk.Button(control_row1, text="Save Matrix", command=self.save_matrix)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.load_button = ttk.Button(control_row1, text="Load Matrix", command=self.load_matrix)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Row 2: Value Selection, Fill Buttons *** (Buttons added here) ***
        ttk.Label(control_row2, text="Paint Value:").pack(side=tk.LEFT, padx=5)
        for i in range(4):
            rb = ttk.Radiobutton(control_row2, text=str(i), variable=self.selected_value, value=i)
            rb.pack(side=tk.LEFT, padx=5)

        # *** NEW: Fill Buttons ***
        self.fill_zero_button = ttk.Button(control_row2, text="Fill 0", command=self.fill_with_zero)
        self.fill_zero_button.pack(side=tk.LEFT, padx=(20, 5)) # Add extra padding before
        self.fill_one_button = ttk.Button(control_row2, text="Fill 1", command=self.fill_with_one)
        self.fill_one_button.pack(side=tk.LEFT, padx=5)
        # *** END NEW ***

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Click or Drag (0/1) to paint.")
        self.status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initial Grid Creation ---
        self.create_or_resize_grid()

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ----- Get current time for status -----
    def _get_timestamp(self):
        # Use current date from system, format as HH:MM:SS
        # Note: Using system time, not explicitly the Auckland time unless system is set to it.
        # For explicit timezone handling, would need pytz or similar.
        # As of Python 3.9+, zoneinfo is built-in.
        try:
            # from zoneinfo import ZoneInfo # Requires Python 3.9+
            # from datetime import datetime
            # dt_now = datetime.now(ZoneInfo("Pacific/Auckland"))
            # return dt_now.strftime("%H:%M:%S")
            # Fallback for older Python:
            return time.strftime("%H:%M:%S")
        except Exception:
            return time.strftime("%H:%M:%S") # Fallback if zoneinfo fails


    def create_or_resize_grid(self):
        try:
            new_rows = self.rows.get()
            new_cols = self.cols.get()
            if new_rows <= 0 or new_cols <= 0:
                raise ValueError("Rows and cols must be positive.")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid Size", f"Enter positive integers for rows/cols.\n{e}")
            current_shape = self.matrix_data.shape if self.matrix_data is not None else (0,0)
            self.rows.set(current_shape[0] if current_shape[0] > 0 else DEFAULT_ROWS)
            self.cols.set(current_shape[1] if current_shape[1] > 0 else DEFAULT_COLS)
            return

        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.cell_buttons = []
        self.is_dragging = False

        self.matrix_data = np.zeros((new_rows, new_cols), dtype=int)
        self.cell_buttons = [[None for _ in range(new_cols)] for _ in range(new_rows)]
        button_width = max(2, CELL_SIZE // 10)
        button_height = max(1, CELL_SIZE // 20)

        for r in range(new_rows):
            for c in range(new_cols):
                val = self.matrix_data[r, c]
                color = VALUE_COLORS.get(val, "#CCCCCC")
                btn = tk.Button(
                    self.grid_frame, text=str(val), width=button_width, height=button_height,
                    bg=color, fg="black", relief=tk.RAISED, bd=1,
                )
                btn.bind("<ButtonPress-1>", lambda event, row=r, col=c: self.handle_button_press(event, row, col))
                btn.bind("<ButtonPress-3>", lambda event, row=r, col=c: self.handle_right_button_press(event, row, col))
                btn.bind("<Enter>", lambda event, row=r, col=c: self.handle_button_enter(event, row, col))
                btn.grid(row=r, column=c, padx=0, pady=0, sticky="nsew")
                self.grid_frame.grid_rowconfigure(r, weight=1, minsize=CELL_SIZE)
                self.grid_frame.grid_columnconfigure(c, weight=1, minsize=CELL_SIZE)
                self.cell_buttons[r][c] = btn

        self.status_var.set(f"{self._get_timestamp()}: Grid {new_rows}x{new_cols}. Click/Drag(0/1) paint.")
        self.on_frame_configure(None) # Update scroll region after creating the grid


    def _update_single_button_ui(self, r, c, value):
        if 0 <= r < len(self.cell_buttons) and 0 <= c < len(self.cell_buttons[0]):
            button = self.cell_buttons[r][c]
            if button:
                color = VALUE_COLORS.get(value, "#CCCCCC")
                button.config(text=str(value), bg=color)

    # ----- Event Handlers -----
    def handle_button_press(self, event, r, c):
        selected_val = self.selected_value.get()
        if selected_val in [0, 1]:
            #self.is_dragging = True
            self.cell_clicked(r, c, update_ui_fully=False)
            self.status_var.set(f"{self._get_timestamp()}: Painting {selected_val}...")
        else:
            self.cell_clicked(r, c, update_ui_fully=True)

    def handle_right_button_press(self, event, r, c):
        selected_val = self.selected_value.get()
        if selected_val in [0, 1]:
            self.is_dragging = not self.is_dragging
            if self.is_dragging:
                self.cell_clicked(r, c, update_ui_fully=False)
                self.status_var.set(f"{self._get_timestamp()}: Start Drag Painting {selected_val}...")
            else:
                self.status_var.set(f"{self._get_timestamp()}: Drag Painting stopped. Ready.")

    def handle_right_button_release(self, event):
        pass

    def handle_button_enter(self, event, r, c):
        if self.is_dragging:
            selected_val = self.selected_value.get()
            if selected_val in [0, 1]:
                self.status_var.set(f"{self._get_timestamp()}: Dragging {r} {c} Right-click to stop dragging.")
                if 0 <= r < self.matrix_data.shape[0] and 0 <= c < self.matrix_data.shape[1]:
                    if self.matrix_data[r, c] != selected_val:
                        self.cell_clicked(r, c, update_ui_fully=False)
            else:
                self.is_dragging = False
                self.status_var.set(f"{self._get_timestamp()}: Drag stopped. Ready.")

    def handle_button_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            self.status_var.set(f"{self._get_timestamp()}: Painting finished. Ready.")


    def _on_mousewheel(self, event):
        """Handles vertical scrolling with the mouse wheel."""
        if event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    def _on_shift_mousewheel(self, event):
        """Handles horizontal scrolling with the mouse wheel (when Shift is pressed)."""
        if event.delta > 0:
            self.canvas.xview_scroll(-1, "units")
        else:
            self.canvas.xview_scroll(1, "units")


    # ----- Core Logic -----
    def cell_clicked(self, r, c, update_ui_fully=True):
        new_value = self.selected_value.get()
        rows, cols = self.matrix_data.shape
        if not (0 <= r < rows and 0 <= c < cols): return

        cleared_value = None
        if new_value == 2:
            locations_to_clear = self.matrix_data == 2
            if np.any(locations_to_clear):
                self.matrix_data[locations_to_clear] = 0
                cleared_value = 2
                update_ui_fully = True
        elif new_value == 3:
            locations_to_clear = self.matrix_data == 3
            if np.any(locations_to_clear):
                self.matrix_data[locations_to_clear] = 0
                cleared_value = 3
                update_ui_fully = True

        # Only update if value actually changes or if full update needed
        if self.matrix_data[r, c] != new_value or update_ui_fully:
            self.matrix_data[r, c] = new_value
            if update_ui_fully:
                self.update_grid_display()
            else:
                self._update_single_button_ui(r, c, new_value)

            if not self.is_dragging: # Update status only for single clicks here
                status_msg = f"Set cell ({r},{c}) to {new_value}"
                if cleared_value is not None: status_msg += f". Cleared {cleared_value}s."
                self.status_var.set(f"{self._get_timestamp()}: {status_msg}")


    def update_grid_display(self):
        rows, cols = self.matrix_data.shape
        if len(self.cell_buttons) != rows or (rows > 0 and len(self.cell_buttons) and len(self.cell_buttons[0]) != cols):
            print("Warning: Mismatch data/button grid. Recreating.")
            # self.create_or_resize_grid() # Be careful with recursion here
            return # Avoid potential infinite loop if create fails repeatedly

        for r in range(rows):
            for c in range(cols):
                if r < len(self.cell_buttons) and c < len(self.cell_buttons[r]):
                    button = self.cell_buttons[r][c]
                    if button:
                        val = self.matrix_data[r, c]
                        color = VALUE_COLORS.get(val, "#CCCCCC")
                        button.config(text=str(val), bg=color)
        self.on_frame_configure(None) # Update scroll region after updating the display


    # *** NEW: Fill Methods ***
    def fill_with_zero(self):
        """Sets all matrix cells to 0 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            self.matrix_data.fill(0)
            self.update_grid_display()
            self.status_var.set(f"{self._get_timestamp()}: Matrix filled with 0 ({self.matrix_data.shape[0]}x{self.matrix_data.shape[1]})")
            print("Matrix filled with 0.") # Debug/Console output
        else:
            self.status_var.set(f"{self._get_timestamp()}: Create a grid first.")

    def fill_with_one(self):
        """Sets all matrix cells to 1 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            self.matrix_data.fill(1)
            self.update_grid_display()
            self.status_var.set(f"{self._get_timestamp()}: Matrix filled with 1 ({self.matrix_data.shape[0]}x{self.matrix_data.shape[1]})")
            print("Matrix filled with 1.") # Debug/Console output
        else:
            self.status_var.set(f"{self._get_timestamp()}: Create a grid first.")
    # *** END NEW ***

    # --- Save/Load Methods (Unchanged) ---
    def save_matrix(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")], title="Save Matrix As"
        )
        if not filepath: self.status_var.set(f"{self._get_timestamp()}: Save cancelled."); return
        try:
            np.save(filepath, self.matrix_data)
            self.status_var.set(f"{self._get_timestamp()}: Matrix saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save matrix:\n{e}")
            self.status_var.set(f"{self._get_timestamp()}: Save failed.")

    def load_matrix(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")], title="Load Matrix"
        )
        if not filepath: self.status_var.set(f"{self._get_timestamp()}: Load cancelled."); return
        try:
            loaded_data = np.load(filepath)
            if not isinstance(loaded_data, np.ndarray): raise TypeError("Not a NumPy array.")
            if loaded_data.ndim != 2: raise ValueError("Array is not 2-dimensional.")
            if not np.all(np.isin(loaded_data, [0, 1, 2, 3])): raise ValueError("Invalid values (0-3).")

            #self.matrix_data = loaded_data.astype(int) #TJH: create_or_resize_grid() resets matrix_data so set matrix_data later
            new_rows, new_cols = loaded_data.shape   #self.matrix_data.shape
            self.rows.set(new_rows)
            self.cols.set(new_cols)
            self.create_or_resize_grid()
            self.matrix_data = loaded_data.astype(int)  #TJH: create_or_resize_grid() resets matrix_data so redo here
            self.update_grid_display() # Ensure loaded data visuals are applied
            self.status_var.set(f"{self._get_timestamp()}: Matrix loaded from {os.path.basename(filepath)}")
        except FileNotFoundError:
            messagebox.showerror("Load Error", f"File not found:\n{filepath}")
            self.status_var.set(f"{self._get_timestamp()}: Load failed: File not found.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load or process matrix:\n{e}")
            self.status_var.set(f"{self._get_timestamp()}: Load failed.")


def main():
    try: import numpy
    except ImportError:
        print("Error: NumPy required.", file=sys.stderr); print("Install: pip install numpy", file=sys.stderr)
        try: root = tk.Tk(); root.withdraw(); messagebox.showerror("Missing Dependency", "NumPy required.\npip install numpy");
        except Exception: pass
        sys.exit(1)

    root = tk.Tk()
    app = MatrixEditorApp(root)
    root.minsize(650, 400) # Increased min width slightly for new buttons
    root.mainloop()

if __name__ == "__main__":
    main()