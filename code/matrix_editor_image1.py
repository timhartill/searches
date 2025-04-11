"""
Numpy Matrix Editor with Image Canvas for creating/editting spatial grid problems

Note that on Ubuntu tkinter isnt installed by default and you will get a
"module not found" error. You can install tkinter with:

$ sudo apt update
$ sudo apt install python3-tk

after that, importing tkinter should work correctly

usage: python matrix_editor_image1.py

Better performance on large matrices than the grid of buttons in matrix_editor2.py

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PhotoImage
import numpy as np
import sys
import os
import time # Import time for timestamp in status

# --- Configuration ---
DEFAULT_ROWS = 1000 # Increased default for testing performance
DEFAULT_COLS = 1000 # Increased default for testing performance
CELL_SIZE = 4   # Can be smaller now, image handles rendering
VALUE_COLORS = {
    0: "#FFFFFF",  # White = empty/traversible
    1: "#ADD8E6",  # Light Blue = wall
    2: "#90EE90",  # Light Green = start
    3: "#FFB6C1",  # Light Pink = goal
    -1: "#CCCCCC", # Default/Error color
}

BUTTON_TEXT = {
0: "EMPTY",
1: "OBSTACLE",
2: "START",
3: "GOAL",
}

# --- End Configuration ---

class MatrixEditorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Matrix Editor (Image Canvas)") # Updated title

        # --- Data ---
        self.rows = tk.IntVar(value=DEFAULT_ROWS)
        self.cols = tk.IntVar(value=DEFAULT_COLS)
        self.matrix_data = np.zeros((DEFAULT_ROWS, DEFAULT_COLS), dtype=int)
        self.selected_value = tk.IntVar(value=0)
        self.is_dragging = False
        self.photo_image = None # To store the PhotoImage object
        self.image_item = None  # To store the canvas image item ID
        self.start_pos = None   # To store current start (row, col)
        self.goal_pos = None    # To store current goal (row, col)

        # --- UI Frames ---
        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # --- Scrollable Canvas for Image ---
        self.canvas_frame = ttk.Frame(master) # Frame to hold canvas and scrollbars
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="grey") # Canvas to display the image
        self.vsb = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # --- Bind mouse events to the canvas ---
        self.canvas.bind("<ButtonPress-1>", self.handle_canvas_press)
        self.canvas.bind("<B1-Motion>", self.handle_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_canvas_release)
        # Bind mouse wheel events to the canvas for scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel) # Windows/Mac horizontal
        self.canvas.bind("<Shift-Button-4>", self._on_shift_mousewheel)   # Linux horizontal scroll up
        self.canvas.bind("<Shift-Button-5>", self._on_shift_mousewheel)   # Linux horizontal scroll down

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

        # Row 2: Value Selection, Fill Buttons
        ttk.Label(control_row2, text="Paint Value:").pack(side=tk.LEFT, padx=5)
        for i in range(4):
            rb = ttk.Radiobutton(control_row2, text=str(i)+' '+BUTTON_TEXT[i], variable=self.selected_value, value=i, command=self._on_value_select)
            rb.pack(side=tk.LEFT, padx=5)

        self.fill_zero_button = ttk.Button(control_row2, text="Fill 0", command=self.fill_with_zero)
        self.fill_zero_button.pack(side=tk.LEFT, padx=(20, 5))
        self.fill_one_button = ttk.Button(control_row2, text="Fill 1", command=self.fill_with_one)
        self.fill_one_button.pack(side=tk.LEFT, padx=5)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Click or Drag to paint.")
        self.status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initial Grid Creation ---
        self.create_or_resize_grid()

    # ----- Get current time for status -----
    def _get_timestamp(self):
        try:
            return time.strftime("%H:%M:%S")
        except Exception:
            return "" # Fallback

    # ----- Grid/Image Creation and Update -----
    def create_or_resize_grid(self):
        """Creates a new matrix and updates the canvas image."""
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

        self.is_dragging = False
        self.matrix_data = np.zeros((new_rows, new_cols), dtype=int)
        self.start_pos = None
        self.goal_pos = None
        self.update_image_display() # Generate and display the new image
        self.status_var.set(f"{self._get_timestamp()}: Created {new_rows}x{new_cols} grid. Ready.")

    def update_image_display(self):
        """Generates a PhotoImage from the matrix_data and displays it on the canvas."""
        rows, cols = self.matrix_data.shape
        width = cols * CELL_SIZE
        height = rows * CELL_SIZE

        # Create a new PhotoImage
        self.photo_image = PhotoImage(width=width, height=height)

        # Fill the image with colors based on matrix data
        # Using put to set blocks of color - more efficient than pixel by pixel for solid cells
        for r in range(rows):
            for c in range(cols):
                val = self.matrix_data[r, c]
                color = VALUE_COLORS.get(val, VALUE_COLORS[-1]) # Use default color if value invalid
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                # The syntax {col1 col2 ...} repeats the color for the line segment
                # Need to create a row string and then join rows with spaces
                # Simpler: Use put with coordinates for blocks
                self.photo_image.put(color, to=(x0, y0, x1, y1))

        # Update the canvas
        if self.image_item:
            self.canvas.delete(self.image_item) # Remove old image item

        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        # Update the scroll region to match the new image size
        self.canvas.config(scrollregion=(0, 0, width, height))
        # print(f"Updated image display: {rows}x{cols}") # Debug

    def _update_image_cell(self, r, c, value):
        """Updates the color of a single cell on the existing PhotoImage."""
        if not self.photo_image or not (0 <= r < self.matrix_data.shape[0] and 0 <= c < self.matrix_data.shape[1]):
            # print(f"Skipping cell update: r={r}, c={c}") # Debug
            return # Image not ready or out of bounds

        color = VALUE_COLORS.get(value, VALUE_COLORS[-1])
        x0 = c * CELL_SIZE
        y0 = r * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        try:
            self.photo_image.put(color, to=(x0, y0, x1, y1))
            # print(f"Updated cell ({r},{c}) to {value} ({color})") # Debug
        except tk.TclError as e:
            print(f"Error updating cell ({r},{c}) with color {color}: {e}") # Debug Tcl errors


    # ----- Event Handlers -----
    def _get_cell_coords(self, event):
        """Converts canvas event coordinates to matrix row and column."""
        # Adjust for canvas scrolling
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        # Calculate row and column
        r = int(canvas_y // CELL_SIZE)
        c = int(canvas_x // CELL_SIZE)
        return r, c

    def handle_canvas_press(self, event):
        """Handles the start of a click or drag operation on the canvas."""
        r, c = self._get_cell_coords(event)
        rows, cols = self.matrix_data.shape
        if 0 <= r < rows and 0 <= c < cols:
            self.is_dragging = True
            paint_value = self.selected_value.get()
            self.status_var.set(f"{self._get_timestamp()}: Painting {paint_value} at ({r},{c})...")
            self.update_cell(r, c, is_drag=False) # Initial click update
        else:
            self.is_dragging = False # Click outside bounds

    def handle_canvas_drag(self, event):
        """Handles mouse movement while the button is pressed (dragging)."""
        if not self.is_dragging:
            return
        r, c = self._get_cell_coords(event)
        rows, cols = self.matrix_data.shape
        if 0 <= r < rows and 0 <= c < cols:
            # Update status less frequently during drag if needed
            # self.status_var.set(f"{self._get_timestamp()}: Dragging over ({r},{c})...")
            self.update_cell(r, c, is_drag=True)
        # else: # Optional: Stop dragging if cursor leaves bounds?
        #     self.is_dragging = False
        #     self.status_var.set(f"{self._get_timestamp()}: Drag ended (out of bounds).")

    def handle_canvas_release(self, event):
        """Handles the release of the mouse button."""
        if self.is_dragging:
            self.is_dragging = False
            # Update status after drag finishes
            r, c = self._get_cell_coords(event) # Get coords at release point
            self.status_var.set(f"{self._get_timestamp()}: Painting finished near ({r},{c}). Ready.")
        else:
             self.status_var.set(f"{self._get_timestamp()}: Ready.")

    def _on_value_select(self):
        """Called when a radio button for paint value is selected."""
        # Provides immediate feedback if needed, e.g., update status
        val = self.selected_value.get()
        self.status_var.set(f"{self._get_timestamp()}: Selected paint value: {val}. Ready.")

    def _on_mousewheel(self, event):
        """Handles vertical scrolling with the mouse wheel."""
        # Determine scroll direction (platform differences)
        if event.num == 4: # Linux scroll up
            delta = -1
        elif event.num == 5: # Linux scroll down
            delta = 1
        else: # Windows/Mac
             delta = -1 if event.delta > 0 else 1 # Inverted delta for yview_scroll

        self.canvas.yview_scroll(delta, "units")
        return "break" # Prevents default scroll behavior if needed

    def _on_shift_mousewheel(self, event):
        """Handles horizontal scrolling with Shift + mouse wheel."""
        if event.num == 4: # Linux scroll left (Shift+Up)
            delta = -1
        elif event.num == 5: # Linux scroll right (Shift+Down)
            delta = 1
        else: # Windows/Mac
            delta = -1 if event.delta > 0 else 1

        self.canvas.xview_scroll(delta, "units")
        return "break"

    # ----- Core Logic -----
    def update_cell(self, r, c, is_drag=False):
        """Updates the matrix data and the corresponding image cell."""
        new_value = self.selected_value.get()
        rows, cols = self.matrix_data.shape

        # Bounds check
        if not (0 <= r < rows and 0 <= c < cols):
            # print(f"Out of bounds: ({r},{c})") # Debug
            return

        current_value = self.matrix_data[r, c]

        # If the value isn't changing, don't do anything (optimization for dragging)
        if current_value == new_value:
            return

        cell_to_clear = None # Store coords of cell to set back to 0

        # Handle unique start (2) and goal (3) points
        if new_value == 2: # Placing start
            if self.start_pos and self.start_pos != (r, c):
                # Clear previous start if it exists and is different
                prev_r, prev_c = self.start_pos
                if 0 <= prev_r < rows and 0 <= prev_c < cols:
                    self.matrix_data[prev_r, prev_c] = 0
                    cell_to_clear = (prev_r, prev_c) # Mark for UI update
            self.start_pos = (r, c)
        elif new_value == 3: # Placing goal
            if self.goal_pos and self.goal_pos != (r, c):
                # Clear previous goal if it exists and is different
                prev_r, prev_c = self.goal_pos
                if 0 <= prev_r < rows and 0 <= prev_c < cols:
                    self.matrix_data[prev_r, prev_c] = 0
                    cell_to_clear = (prev_r, prev_c) # Mark for UI update
            self.goal_pos = (r, c)
        elif current_value == 2: # Overwriting start
             self.start_pos = None
        elif current_value == 3: # Overwriting goal
             self.goal_pos = None


        # Update the matrix data
        self.matrix_data[r, c] = new_value

        # Update the image display for the changed cell(s)
        if cell_to_clear:
            self._update_image_cell(cell_to_clear[0], cell_to_clear[1], 0)

        self._update_image_cell(r, c, new_value)

        # Update status bar only on initial click, not during drag for performance
        if not is_drag:
            status_msg = f"Set cell ({r},{c}) to {new_value}"
            if cell_to_clear: status_msg += f". Cleared previous {self.matrix_data[cell_to_clear[0], cell_to_clear[1]]}."
            # self.status_var.set(f"{self._get_timestamp()}: {status_msg}") # Can be noisy, keep simple status


    # ----- Fill Methods -----
    def fill_with_zero(self):
        """Sets all matrix cells to 0 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            self.matrix_data.fill(0)
            self.start_pos = None # Clear start/goal positions
            self.goal_pos = None
            self.update_image_display() # Redraw the whole image
            self.status_var.set(f"{self._get_timestamp()}: Matrix filled with 0 ({self.matrix_data.shape[0]}x{self.matrix_data.shape[1]})")
            print("Matrix filled with 0.")
        else:
            self.status_var.set(f"{self._get_timestamp()}: Create a grid first.")

    def fill_with_one(self):
        """Sets all matrix cells to 1 and updates the UI."""
        if self.matrix_data is not None and self.matrix_data.size > 0:
            # Preserve start/goal if they exist? Or fill everything?
            # Current: Fills everything, potentially overwriting start/goal
            self.matrix_data.fill(1)
            # If start/goal should be preserved, re-apply them:
            # if self.start_pos: self.matrix_data[self.start_pos] = 2
            # if self.goal_pos: self.matrix_data[self.goal_pos] = 3
            # Let's assume fill means fill *everything* for simplicity now.
            self.start_pos = None
            self.goal_pos = None
            self.update_image_display() # Redraw the whole image
            self.status_var.set(f"{self._get_timestamp()}: Matrix filled with 1 ({self.matrix_data.shape[0]}x{self.matrix_data.shape[1]})")
            print("Matrix filled with 1.")
        else:
            self.status_var.set(f"{self._get_timestamp()}: Create a grid first.")

    # ----- Save/Load Methods -----
    def save_matrix(self):
        """Saves the current matrix data to a .npy file."""
        if self.matrix_data is None or self.matrix_data.size == 0:
             messagebox.showwarning("Save Error", "No matrix data to save.")
             self.status_var.set(f"{self._get_timestamp()}: Save failed: No data.")
             return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")], title="Save Matrix As"
        )
        if not filepath:
            self.status_var.set(f"{self._get_timestamp()}: Save cancelled."); return
        try:
            np.save(filepath, self.matrix_data)
            self.status_var.set(f"{self._get_timestamp()}: Matrix saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save matrix:\n{e}")
            self.status_var.set(f"{self._get_timestamp()}: Save failed.")

    def load_matrix(self):
        """Loads matrix data from a .npy file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")], title="Load Matrix"
        )
        if not filepath:
            self.status_var.set(f"{self._get_timestamp()}: Load cancelled."); return
        try:
            loaded_data = np.load(filepath)
            if not isinstance(loaded_data, np.ndarray): raise TypeError("File is not a NumPy array.")
            if loaded_data.ndim != 2: raise ValueError("Loaded array is not 2-dimensional.")
            # Allow any integer values on load, but maybe validate 0-3?
            # For now, accept any integer array.
            # if not np.all(np.isin(loaded_data, [0, 1, 2, 3])): raise ValueError("Invalid values (only 0-3 allowed).") # Optional validation

            self.matrix_data = loaded_data.astype(int)
            new_rows, new_cols = self.matrix_data.shape
            self.rows.set(new_rows)
            self.cols.set(new_cols)

            # Find start/goal positions in the loaded data
            start_indices = np.argwhere(self.matrix_data == 2)
            goal_indices = np.argwhere(self.matrix_data == 3)
            self.start_pos = tuple(start_indices[0]) if len(start_indices) > 0 else None
            self.goal_pos = tuple(goal_indices[0]) if len(goal_indices) > 0 else None
            # If multiple starts/goals exist, only the first found is stored.
            # Consider adding logic to handle/warn about multiple starts/goals if needed.

            self.update_image_display() # Update the image display
            self.status_var.set(f"{self._get_timestamp()}: Matrix loaded from {os.path.basename(filepath)} ({new_rows}x{new_cols})")
        except FileNotFoundError:
            messagebox.showerror("Load Error", f"File not found:\n{filepath}")
            self.status_var.set(f"{self._get_timestamp()}: Load failed: File not found.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load or process matrix:\n{e}")
            self.status_var.set(f"{self._get_timestamp()}: Load failed.")


def main():
    # Check for NumPy dependency
    try:
        import numpy
    except ImportError:
        error_msg = "Error: NumPy is required for this application.\nPlease install it using: pip install numpy"
        print(error_msg, file=sys.stderr)
        try:
            # Attempt to show a GUI error message if Tkinter is available
            root_check = tk.Tk()
            root_check.withdraw() # Hide the main window
            messagebox.showerror("Missing Dependency", error_msg)
            root_check.destroy()
        except tk.TclError:
            pass # Cannot show GUI error if Tkinter itself fails
        except Exception as e:
            print(f"Could not display error message box: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the application
    root = tk.Tk()
    app = MatrixEditorApp(root)
    root.minsize(1000, 1000) # Adjust minimum size as needed
    root.mainloop()

if __name__ == "__main__":
    main()
