use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList}; 
use std::collections::HashMap;

// Unused - using RustDict below resulted in only slightly less memory used but ran slower than the python namedtuple version.

/// Manhattan heuristic - not used as the python version is slightly faster likely because of the calling overhead.
#[pyfunction]
fn heuristic_manhattan(
    state_bytes: Vec<u8>,
    target_positions: HashMap<u8, u8>,
    max_cols: Option<usize>,
    degradation: Option<usize>,
    make_heuristic_inadmissable: Option<bool>,
) -> PyResult<usize> {
    let max_cols = max_cols.unwrap_or(4);
    let degradation = degradation.unwrap_or(0);
    let make_heuristic_inadmissable = make_heuristic_inadmissable.unwrap_or(false);
    let mut distance = 0;
    let mut multiplier = 1;

    let mut ignored_tiles: Vec<u8> = Vec::new();
    for i in 0..=degradation {
        ignored_tiles.push(i as u8);
    }
    
    for (i, &tile) in state_bytes.iter().enumerate() {
        if !ignored_tiles.contains(&tile) {
            let current_pos_row = i / max_cols;
            let current_pos_col = i % max_cols;

            if let Some(&goal_idx) = target_positions.get(&tile) {
                let goal_pos_row = goal_idx as usize / max_cols;
                let goal_pos_col = goal_idx as usize % max_cols;

                if make_heuristic_inadmissable {
                    multiplier = i + 1;
                }

                let manhattan_dist = (current_pos_row as isize - goal_pos_row as isize).abs()
                    + (current_pos_col as isize - goal_pos_col as isize).abs();
                
                distance += multiplier * (manhattan_dist as usize);
            }
        }
    }
    Ok(distance)
}


/// A struct to hold the node data, exposed to Python as a class.
/// The `clone` attribute is needed so we can return copies from __getitem__.
/// `get_all` and `set_all` automatically create getters and setters for the fields.
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug)] 
struct NodeData {
    g: f32,
    h: f32,
    parent: Option<Vec<u8>>, // Changed to Option<Vec<u8>> for nullable parent
}


#[pymethods]
impl NodeData {
    /// A constructor for creating NodeData objects directly from Python if needed.
    #[new]
    fn new(g: f32, h: f32, parent: Option<Vec<u8>>) -> Self {
        NodeData { g, h, parent } 
    }
    
    /// Provides a nice string representation for printing in Python.
    fn __repr__(&self) -> String {
        let parent_str = match &self.parent {
            Some(p) => format!("'{}'", String::from_utf8_lossy(p)),
            None => "None".to_string(),
        };
        format!("<NodeData g={}, h={}, parent={}>", self.g, self.h, parent_str)
    }
}


/// The main dictionary-like class that stores keys and NodeData values.
#[pyclass]
struct RustDict {
    data: HashMap<Vec<u8>, NodeData>, // Changed HashMap key from String to Vec<u8>
}


#[pymethods]
impl RustDict {
    #[new]
    fn new() -> Self {
        RustDict {
            data: HashMap::new(),
        }
    }

    /// Retrieves a NodeData object. PyO3 handles the conversion to a Python object.
    fn __getitem__(&self, key: Vec<u8>) -> PyResult<NodeData> { // Takes &PyBytes for key
        self.data
            .get(&key) 
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key))
            
    }

    /// Create/update an entry directly from a NodeData object
    fn __setitem__(&mut self, key: Vec<u8>, value: NodeData) -> PyResult<()> {
        self.data.insert(key, value);
        Ok(())
    }    

    /// Takes a Python dictionary as input and converts it into a Rust NodeData struct for storage.
    fn add_or_update(&mut self, key: Vec<u8>, value: &Bound<'_, PyDict>) -> PyResult<()> { 
        let g: f32 = value
            .get_item("g")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Value dict must contain 'g'"))?
            .extract()?;
        let h: f32 = value
            .get_item("h")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Value dict must contain 'h'"))?
            .extract()?;
        let parent: Option<Vec<u8>> = value 
            .get_item("parent")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Value dict must contain 'parent'"))?
            .extract()?;        
        let node = NodeData { g, h, parent };
        self.data.insert(key, node); 
        Ok(())
    }
    
    /// Returns the number of items in the dictionary.
    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Efficiently gets a NodeData object by key, returning None if the key does not exist.
    /// This is the equivalent of Python's `dict.get(key, None)`.
    fn get(&self, key: Vec<u8>) -> PyResult<Option<NodeData>> { 
        Ok(self.data.get(&key).cloned()) 
    }

    /// Enables Python's `key in dict` syntax.
    /// Returns `True` if the key exists, `False` otherwise.
    fn __contains__(&self, key: Vec<u8>) -> bool { 
        self.data.contains_key(&key) 
    }

    /// Returns a new Python list containing all the keys in the dictionary.
    fn keys<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for key in self.data.keys() {
            list.append(key.clone())?;
        }
        Ok(list)
    }

}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the `heuristic_manhattan` function
    m.add_function(wrap_pyfunction!(heuristic_manhattan, m)?)?;
    // Add the `NodeData` class
    m.add_class::<NodeData>()?;
    // Add the `RustDict` class
    m.add_class::<RustDict>()?;
    Ok(())
 }



