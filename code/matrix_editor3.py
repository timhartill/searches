""" Numpy matrix editor using PhotoImage for faster rendering """

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
        self.master.title("Matrix Editor (0-3) - Drag Paint & Fill (PhotoImage)")

        # --- Data ---
        self.rows = tk.IntVar(value=DEFAULT_ROWS)
        self.cols = tk.IntVar(value=DEFAULT_COLS)
        self.matrix_data = np.zeros((DEFAULT_ROWS, DEFAULT_COLS), dtype=int)
        self.selected_value = tk.IntVar(value=0)
        self.is_dragging = False
        self.grid_image = None
        self.grid_image_id = None

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

        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.handle_button_press)
        self.canvas.bind("<B1-Motion>", self.handle_button_enter) # Drag with left button
        self.canvas.bind("<ButtonRelease-1>", self.handle_button_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # For vertical scrolling
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel) # For horizontal scrolling

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

    def on_canvas_configure(self, event):
        """Update the scroll region to encompass the image size."""
        if self.grid_image:
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

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

    # ----- Get current time for status -----
    def _get_timestamp(self):
        return time.strftime("%H:%M:%S")

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

        self.matrix_data = np.zeros((new_rows, new_cols), dtype=int)
        self._draw_grid_image()
        self.status_var.set(f"{self._get_timestamp()}: Grid {new_rows}x{new_cols}. Click/Drag(0/1) paint.")

    def _draw_grid_image(self):
        rows, cols = self.matrix_data.shape
        width = cols * CELL_SIZE
        height = rows * CELL_SIZE
        self.grid_image = tk.PhotoImage(width=width, height=height)

        for r in range(rows):
            for c in range(cols):
                val = self.matrix_data[r, c]
                color = VALUE_COLORS.get(val, "#CCCCCC")
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = (c + 1) * CELL_SIZE - 1, (r + 1) * CELL_SIZE - 1
                self.grid_image.put(color, to=(x1, y1, x2, y2))

        # Display the image on the canvas
        if self.grid_image_id:
            self.canvas.itemconfig(self.grid_image_id, image=self.grid_image)
        else:
            self.grid_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.grid_image)
        self.on_canvas_configure(None)

    def _update_cell(self, r, c, value):
        if 0 <= r < self.matrix_data.shape[0] and 0 <= c < self.matrix_data.shape[1]:
            if self.matrix_data[r, c] != value:
                self.matrix_data[r, c] = value
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = (c + 1) * CELL_SIZE - 1, (r + 1) * CELL_SIZE - 1
                color = VALUE_COLORS.get(value, "#CCCCCC")
                self.grid_image.put(color, to=(x1, y1, x2, y2))

    # ----- Event Handlers -----
    def handle_button_press(self, event):
        selected_val = self.selected_value.get()
        if selected_val in [0, 1]:
            self.is_dragging = True
            r = event.y // CELL_SIZE
            c = event.x // CELL_SIZE
            self._update_cell(r, c, selected_val)
            self.status_var.set(f"{self._get_timestamp()}: Painting {selected_val}...")
        else:
            r = event.y // CELL_SIZE
            c = event.x // CELL_SIZE
            self.cell_clicked(r, c)

    def handle_button_enter(self, event):
        if self.is_dragging:
            selected_val = self.selected_value.get()
            if selected_val in [0, 1]:
                r = event.y // CELL_SIZE
                c = event.x // CELL_SIZE
                self._update_cell(r, c, selected_val)
            else:
                self.is_dragging = False
                self.status_var.set(f"{self._get_timestamp()}: Drag stopped. Ready.")

    def handle_button_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            self.status_var.set(f"{self._get_timestamp()}: Painting finished. Ready.")

    # ----- Core Logic -----
    def cell_clicked(self, r, c):
        new_value = self.selected_value.get()
        rows, cols = self.matrix_data.shape
        if not (0 <= r < rows and 0 <= c < cols): return

        cleared_value = None
        if new_value == 2:
            locations_to_clear = self.matrix_data == 2
            if np.any(locations_to_clear):
                self.matrix_data[locations_to_clear] = 0
                cleared_value = 2
                self._draw_grid_image()
        elif new_value == 3:
            locations_to_clear = self.matrix_data == 3
            if np.any(locations_to_clear):
                self.matrix_data[locations_to_clear] = 0
                cleared_value = 3
                self._draw_grid_image()

        # Only update if value actually changes
        if self.matrix_data[r, c] != new_value:
            self.matrix_data[r, c] = new_value
            self._update_cell(r, c, new_value)

            if not self.is_dragging: # Update status only for single clicks here
                status_msg = f"Set cell ({r},{c}) to {new_value}"
                if cleared_value is not None: status_msg += f". Cleared {cleared_value}s."
                self.status_var.set(f"{self._get_timestamp()}: {status_msg}")

    # *** NEW: Fill Methods ***
    def fill_with_zero(self):
        """Sets all matrix cells to 0 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            self.matrix_data.fill(0)
            self._draw_grid_image()
            self.status_var.set(f"{self._get_timestamp()}: Matrix filled with 0 ({self.matrix_data.shape[0]}x{self.matrix_data.shape[1]})")
            print("Matrix filled with 0.") # Debug/Console output
        else:
            self.status_var.set(f"{self._get_timestamp()}: Create a grid first.")

    def fill_with_one(self):
        """Sets all matrix cells to 1 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            self.matrix_data.fill(1)
            self._draw_grid_image()
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

            self.matrix_data = loaded_data.astype(int)
            new_rows, new_cols = self.matrix_data.shape
            self.rows.set(new_rows)
            self.cols.set(new_cols)
            self._draw_grid_image()
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
