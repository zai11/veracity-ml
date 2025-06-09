use std::{any::Any, fmt, sync::{Arc, Mutex, MutexGuard}};

use indexmap::IndexMap;
use ndarray::Array2;

use crate::{data_vector::{downcast_ref, DataVector, TDataVector, TDataVectorExt}, enums::error_types::DataLoaderError};

pub trait TDataMatrix
{
    fn new() -> Self where Self: Sized;
    fn add_column<T: Clone + Send + Sync + 'static>(&mut self, data: Vec<T>, label: Option<&str>) -> Result<(), DataLoaderError>;
    fn add_row(&mut self, row: Vec<&dyn Any>, index: &str) -> Result<(), DataLoaderError>;
    fn set_index(&mut self, index: Vec<&str>) -> Result<(), DataLoaderError>;
    fn from_vec(vec: Vec<Arc<Mutex<dyn TDataVector>>>) -> Result<Self, DataLoaderError> where Self: Sized;
    fn from_ndarray<T: Clone + Send + Sync + 'static>(arr: Array2<T>) -> Result<Self, DataLoaderError> where Self: Sized;
    fn to_ndarray<T: Clone + Send + Sync + 'static>(&self) -> Result<Array2<T>, DataLoaderError>;
    fn is_type_heterogeneous(&self) -> Result<bool, DataLoaderError>;
    fn get_column<T: Clone + Send + Sync + 'static>(&self, column_name: &str) -> Result<DataVector<T>, DataLoaderError>;
    fn get_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError>;
    fn exclude_column(&self, column_name: &str) -> Result<DataMatrix, DataLoaderError>;
    fn exclude_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError>;
    fn nrows(&self) -> usize;
    fn column_at(&self, idx: usize) -> Option<Arc<Mutex<dyn TDataVector>>>;
}

pub struct DataMatrix {
    pub columns: IndexMap<String, Arc<Mutex<dyn TDataVector>>>,
    pub index: Vec<String>
}

impl TDataMatrix for DataMatrix {
    fn new() -> Self {
        DataMatrix { 
            columns: IndexMap::new(), 
            index: Vec::new() 
        }
    }

    fn add_column<T: Clone + Send + Sync + 'static>(&mut self, data: Vec<T>, label: Option<&str>) -> Result<(), DataLoaderError> {
        let len: usize = data.len();

        if self.columns.len() != 0 {
            let first_key: &String = self.columns.keys().next().ok_or(DataLoaderError::NoData)?;
            let first_col: &Arc<Mutex<dyn TDataVector + 'static>> = self.columns.get(first_key).ok_or(DataLoaderError::GenericError("Column not found".into()))?;
            let first_col: MutexGuard<'_, dyn TDataVector + 'static> = first_col.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
        
            if let Some(col) = downcast_ref::<T>(&*first_col) {
                if col.len() != len {
                    return Err(DataLoaderError::RowCountMismatch);
                }
            } else {
                return Err(DataLoaderError::GenericError("Downcast failed".into()));
            }
        }

        let mut column: DataVector<T> = DataVector::from_vec(data)?;

        if let Some(lab) = label {
            for (_, col) in &self.columns {
                let col = col.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
                if col.get_label().as_deref() == Some(lab) {
                    return Err(DataLoaderError::DuplicateLabel);
                }
            }
            column.add_label(lab);
        }

        let label: String = column.get_label().clone().ok_or(DataLoaderError::GenericError("Column didn't have a label".to_string()))?;

        self.columns.insert(label, Arc::from(Mutex::new(column)));
        Ok(())
    }

    fn add_row(&mut self, row: Vec<&dyn Any>, index: &str) -> Result<(), DataLoaderError> {
        if row.len() != self.columns.len() {
            return Err(DataLoaderError::GenericError(format!(
                "Expected {} values, got {}",
                self.columns.len(),
                row.len()
            )));
        }

        for ((_, column_arc), value) in self.columns.iter_mut().zip(row.into_iter()) {
            let mut column = column_arc.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
            column.append(value)?;
        }

        self.index.push(index.to_string());
        Ok(())
    }

    fn set_index(&mut self, index: Vec<&str>) -> Result<(), DataLoaderError> {
        if !self.columns.is_empty() {
            let first_key = self.columns.keys().next().ok_or(DataLoaderError::NoData)?;
            if let Some(first_col) = self.columns.get(first_key) {
                if first_col.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?.len() != index.len() {
                    return Err(DataLoaderError::RowCountMismatch);
                }
            }
        }
    
        let unique_count = index.iter().collect::<std::collections::HashSet<_>>().len();
        if index.len() != unique_count {
            return Err(DataLoaderError::DuplicateIndex);
        }
    
        self.index = index.iter().map(|i| i.to_string()).collect();
        Ok(())
    }

    fn from_vec(vec: Vec<Arc<Mutex<dyn TDataVector>>>) -> Result<Self, DataLoaderError> {
        let mut columns: IndexMap<String, Arc<Mutex<dyn TDataVector>>> = IndexMap::new();
        for data in vec {
            let key = data.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?.get_label().unwrap_or_else(|| columns.len().to_string());
            columns.insert(key, data);
        }
        Ok(DataMatrix {
            columns,
            index: Vec::new()
        })
    }

    fn from_ndarray<T: Clone + Send + Sync + 'static>(arr: Array2<T>) -> Result<Self, DataLoaderError> {
        let mut mat: DataMatrix = DataMatrix::new();
        for row in arr.outer_iter() {
            let data_vector: Vec<T> = row.to_vec();
            mat.add_column(data_vector, None)?;
        }

        Ok(mat)
    }

    fn to_ndarray<T: Clone + Send + Sync + 'static>(&self) -> Result<Array2<T>, DataLoaderError> {
        if let Ok(is_type_heterogenous) = self.is_type_heterogeneous() {
            if is_type_heterogenous {
                return Err(DataLoaderError::HeterogeneousDataTypes)
            }
        }

        if self.columns.len() == 0 {
            return Err(DataLoaderError::NoData);
        }

        let first_label: &String = self.columns.keys().next().ok_or(DataLoaderError::NoData)?;

        let num_rows: usize = self.columns[first_label].lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?.len();

        let mut data: Vec<T> = Vec::with_capacity(num_rows * self.columns.len());

        for row_idx in 0..num_rows {
            for label in self.columns.keys() {
                let col = self.columns[label]
                    .lock()
                    .map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
                let dv = col
                    .as_any()
                    .downcast_ref::<DataVector<T>>()
                    .ok_or_else(|| DataLoaderError::GenericError(format!("Failed to downcast column '{}' to expected type", label)))?;
                let value = dv.iter()?.nth(row_idx).ok_or(DataLoaderError::IndexError(row_idx))?.clone();
                data.push(value);
            }
        }

        Array2::from_shape_vec((num_rows, self.columns.len()), data)
            .map_err(|e: ndarray::ShapeError| DataLoaderError::GenericError(format!("{:#?}", e)))
    }

    fn is_type_heterogeneous(&self) -> Result<bool, DataLoaderError> {
        let first_label: &String = self.columns.keys().next().ok_or(DataLoaderError::NoData)?;
        let first_dtype: &'static str = self.columns[first_label].lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?.dtype();

        for column in self.columns.values() {
            let column: MutexGuard<'_, dyn TDataVector> = column.lock().map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
            if column.dtype() != first_dtype {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn get_column<T: Clone + Send + Sync + 'static>(&self, column_name: &str) -> Result<DataVector<T>, DataLoaderError> {
        let col = self.columns
            .get(column_name)
            .ok_or_else(|| DataLoaderError::GenericError("Column not found".to_string()))?;
    
        let guard = col.lock()
            .map_err(|_| DataLoaderError::GenericError("Mutex poisoned: failed to acquire lock".to_string()))?;
    
        let dv = guard
            .as_any()
            .downcast_ref::<DataVector<T>>()
            .ok_or_else(|| DataLoaderError::GenericError(format!(
                "Failed to downcast column '{}' to expected type",
                column_name
            )))?;
    
        Ok(dv.clone())
    }

    fn get_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError> {
        let mut columns: IndexMap<String, Arc<Mutex<dyn TDataVector>>> = IndexMap::new();
    
        for &name in column_names.iter() {
            let col: &Arc<Mutex<dyn TDataVector>> = self.columns.get(name)
                .ok_or_else(|| DataLoaderError::GenericError("Column not found".to_string()))?;
            columns.insert(name.to_string(), Arc::clone(col));
        }
    
        Ok(DataMatrix {
            columns,
            index: self.index.clone(),
        })
    }

    fn exclude_column(&self, column_name: &str) -> Result<DataMatrix, DataLoaderError> {
        let columns: IndexMap<String, Arc<Mutex<dyn TDataVector>>> = self.columns
            .iter()
            .filter_map(|(name, data_vector)| {
                if name != column_name {
                    Some((name.clone(), Arc::clone(data_vector)))
                } else {
                    None
                }
            })
            .collect();
    
        Ok(DataMatrix {
            columns,
            index: self.index.clone(),
        })
    }

    fn exclude_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError> {
        let excluded_set: std::collections::HashSet<&str> = column_names.into_iter().collect();

        let columns: Vec<Arc<Mutex<dyn TDataVector>>> = self.columns
            .iter()
            .filter_map(|(name, data_vector)| {
                if !excluded_set.contains(name.as_str()) {
                    Some(data_vector.clone())
                } else {
                    None
                }
            })
            .collect();

        DataMatrix::from_vec(columns)
    }

    fn nrows(&self) -> usize {
        self.columns
            .values()
            .next()
            .map(|col| col.lock().unwrap().len())
            .unwrap_or(0)
    }

    fn column_at(&self, idx: usize) -> Option<Arc<Mutex<dyn TDataVector>>> {
        self.columns.values().nth(idx).cloned()
    }
}

impl fmt::Display for DataMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut col_labels: Vec<&String> = self.columns.keys().collect();
        col_labels.sort();

        let mut col_widths: Vec<usize> = col_labels.iter().map(|label| label.len()).collect();
        for (i, label) in col_labels.iter().enumerate() {
            if let Some(col) = self.columns.get(*label) {
                let col = col.lock().map_err(|_| fmt::Error)?;
                if let Some(vec) = col.get_data().downcast_ref::<Vec<String>>() {
                    let max_data_len = vec.iter().map(|s| s.len()).max().unwrap_or(0);
                    col_widths[i] = col_widths[i].max(max_data_len);
                } else if let Some(vec) = col.get_data().downcast_ref::<Vec<f64>>() {
                    let max_data_len = vec.iter().map(|v| format!("{:.3}", v).len()).max().unwrap_or(0);
                    col_widths[i] = col_widths[i].max(max_data_len);
                } else if let Some(vec) = col.get_data().downcast_ref::<Vec<i32>>() {
                    let max_data_len = vec.iter().map(|v| v.to_string().len()).max().unwrap_or(0);
                    col_widths[i] = col_widths[i].max(max_data_len);
                }
            }
        }

        let index_width = self.index.iter().map(|i| i.len()).max().unwrap_or(5).max("Index".len());

        write!(f, "| {:^index_width$} |", "Index", index_width = index_width)?;
        for (label, width) in col_labels.iter().zip(&col_widths) {
            write!(f, " {:^width$} |", label, width = width)?;
        }
        writeln!(f)?;

        write!(f, "+-{:-^index_width$}-+", "", index_width = index_width)?;
        for width in &col_widths {
            write!(f, "-{:-^width$}-+", "", width = width)?;
        }
        writeln!(f)?;

        let num_rows: usize = self
            .columns
            .values()
            .next()
            .map_or(Ok(0), |col| {
                let guard = col.lock().map_err(|_| fmt::Error)?;
                Ok(guard.len())
            })?;
        for row in 0..num_rows {
            let index = self.index.get(row).map(String::as_str).unwrap_or("");
            write!(f, "| {:^index_width$} |", index, index_width = index_width)?;

            for (label, width) in col_labels.iter().zip(&col_widths) {
                let col = self.columns[*label].lock().map_err(|_| fmt::Error)?;
                let dtype = col.dtype();
                match dtype {
                    "bool" => {
                        let vec = col.get_data().downcast_ref::<Vec<bool>>().ok_or(fmt::Error)?;
                        let value = vec.get(row).map(|v| v.to_string()).unwrap_or_else(|| "".to_string());
                        write!(f, " {:^width$} |", value, width = width)?;
                    }
                    "i64" => {
                        let vec = col.get_data().downcast_ref::<Vec<i64>>().ok_or(fmt::Error)?;
                        let value = vec.get(row).map(|v| v.to_string()).unwrap_or_else(|| "".to_string());
                        write!(f, " {:^width$} |", value, width = width)?;
                    }
                    "f64" => {
                        let vec = col.get_data().downcast_ref::<Vec<f64>>().ok_or(fmt::Error)?;
                        let value = vec.get(row).map(|v| format!("{:.3}", v)).unwrap_or_else(|| "".to_string());
                        write!(f, " {:^width$} |", value, width = width)?;
                    }
                    "alloc::string::String" | "String" => {
                        let vec = col.get_data().downcast_ref::<Vec<String>>().ok_or(fmt::Error)?;
                        let value = vec.get(row).map(String::as_str).unwrap_or("");
                        write!(f, " {:^width$} |", value, width = width)?;
                    }
                    _ => writeln!(f, "Unknown dtype '{}'", dtype)?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Clone for DataMatrix {
    fn clone(&self) -> Self {
        let columns = self.columns
            .iter()
            .map(|(k, v)| {
                let cloned_vec: Arc<Mutex<dyn TDataVector>> = {
                    let locked = v.lock().unwrap();
                    let any = locked.as_any();

                    if let Some(vec) = any.downcast_ref::<DataVector<f64>>() {
                        Arc::new(Mutex::new(vec.clone()))
                    } else if let Some(vec) = any.downcast_ref::<DataVector<bool>>() {
                        Arc::new(Mutex::new(vec.clone()))
                    } else if let Some(vec) = any.downcast_ref::<DataVector<String>>() {
                        Arc::new(Mutex::new(vec.clone()))
                    } else {
                        panic!("Unsupported type in DataMatrix column for cloning")
                    }
                };
                (k.clone(), cloned_vec)
            })
            .collect();

        Self {
            columns,
            index: self.index.clone(),
        }
    }
}