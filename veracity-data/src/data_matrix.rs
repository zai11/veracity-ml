use std::{collections::BTreeMap, fmt};

use ndarray::Array2;

use crate::{data_vector::DataVector, enums::error_types::DataLoaderError};

pub struct DataMatrix {
    pub columns: BTreeMap<String, DataVector>,
    pub index: Vec<String>
}

impl DataMatrix {
    pub fn new() -> Self {
        DataMatrix { 
            columns: BTreeMap::new(), 
            index: Vec::new() 
        }
    }

    pub fn add_column<T: Clone + Send + Sync + 'static>(&mut self, data: Vec<T>, label: Option<&str>) -> Result<(), DataLoaderError> {
        let len: usize = data.len();

        if self.columns.len() != 0 {
            if self.columns.get(self.columns.keys().next().expect("No data in DataMatrix")).map_or(false, |first_col: &DataVector| first_col.len != len) {
                return Err(DataLoaderError::RowCountMismatch);
            }
        }

        let mut column: DataVector = DataVector::from_vec(data)?;

        if let Some(lab) = label {
            if self.columns.iter().any(|(_, col)| col.label.as_deref() == Some(lab)) {
                return Err(DataLoaderError::DuplicateLabel);
            }
            column.add_label(lab);
        }

        let label: String = column.label.clone().ok_or(DataLoaderError::GenericError("Column label didn't have a label".to_string()))?;

        self.columns.insert(label, column);
        Ok(())
    }

    pub fn set_index(&mut self, index: Vec<&str>) -> Result<(), DataLoaderError> {
        if self.columns.len() != 0 {
            if self.columns.get(self.columns.keys().next().expect("No data in DataMatrix")).map_or(false, |first_col: &DataVector| first_col.len != index.len()) {
                return Err(DataLoaderError::RowCountMismatch);
            }
        }

        if index.len() != index.iter().collect::<std::collections::HashSet<_>>().len() {
            return Err(DataLoaderError::DuplicateIndex);
        }

        self.index = index.iter().map(|i| i.to_string()).collect();
        Ok(())
    }

    pub fn from_vec(vec: Vec<DataVector>) -> Result<Self, DataLoaderError> {
        let mut columns = BTreeMap::new();
        for data in vec.iter() {
            let key = data.label.clone().unwrap_or(columns.len().to_string());
            columns.insert(key, data.clone());
        }
        Ok(DataMatrix {
            columns,
            index: Vec::new()
        })
    }

    pub fn from_ndarray<T: Clone + Send + Sync + 'static>(arr: Array2<T>) -> Result<Self, DataLoaderError> {
        let mut mat: DataMatrix = DataMatrix::new();
        for row in arr.outer_iter() {
            let data_vector: Vec<T> = row.to_vec();
            mat.add_column(data_vector, None)?;
        }

        Ok(mat)
    }

    pub fn to_ndarray<T: Clone + Send + Sync + 'static>(&self) -> Result<Array2<T>, DataLoaderError> {
        if self.is_type_heterogeneous() {
            return Err(DataLoaderError::HeterogeneousDataTypes)
        }

        if self.columns.len() == 0 {
            return Err(DataLoaderError::NoData);
        }

        let first_label: &String = self.columns.keys().next().expect("No data in DataMatrix");

        let num_rows: usize = self.columns[first_label].len;

        let mut data: Vec<T> = Vec::with_capacity(num_rows * self.columns.len());

        for row_idx in 0..num_rows {
            for label in self.columns.keys() {
                data.push(self.columns[label].iter::<T>()?.nth(row_idx).ok_or(DataLoaderError::IndexError(row_idx))?.clone());
            }
        }

        Array2::from_shape_vec((num_rows, self.columns.len()), data)
            .map_err(|e: ndarray::ShapeError| DataLoaderError::GenericError(format!("{:#?}", e)))
    }

    pub fn is_type_heterogeneous(&self) -> bool {
        let first_label: &String = self.columns.keys().next().expect("No data in DataMatrix");
        let first_dtype: Option<&'static str> = self.columns[first_label].dtype;

        if first_dtype.is_none() {
            return false;
        }

        for &column in &self.columns.values().collect::<Vec<&DataVector>>() {
            if column.dtype != first_dtype {
                return true;
            }
        }

        false
    }

    pub fn get_column(&self, column_name: &str) -> Result<DataVector, DataLoaderError> {
        Ok(self.columns[column_name].clone())
    }

    pub fn get_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError> {
        let columns: Vec<DataVector> = column_names
            .iter()
            .filter_map(|&name| self.columns.get(name).cloned())
            .collect();

        DataMatrix::from_vec(columns)
    }

    pub fn exclude_column(&self, column_name: &str) -> Result<DataMatrix, DataLoaderError> {
        let columns: Vec<DataVector> = self.columns
            .iter()
            .filter_map(|(name, data_vector)| {
                if name != column_name {
                    Some(data_vector.clone())
                } else {
                    None
                }
            })
            .collect();

        DataMatrix::from_vec(columns)
    }

    pub fn exclude_columns(&self, column_names: Vec<&str>) -> Result<DataMatrix, DataLoaderError> {
        let excluded_set: std::collections::HashSet<&str> = column_names.into_iter().collect();

        let columns: Vec<DataVector> = self.columns
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
}

impl fmt::Display for DataMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut col_labels: Vec<&String> = self.columns.keys().collect();
        col_labels.sort();

        let mut col_widths: Vec<usize> = col_labels.iter().map(|label| label.len()).collect();
        for (i, label) in col_labels.iter().enumerate() {
            if let Some(col) = self.columns.get(*label) {
                if let Some(vec) = col.data.downcast_ref::<Vec<String>>() {
                    let max_data_len = vec.iter().map(|s| s.len()).max().unwrap_or(0);
                    col_widths[i] = col_widths[i].max(max_data_len);
                } else if let Some(vec) = col.data.downcast_ref::<Vec<f64>>() {
                    let max_data_len = vec.iter().map(|v| format!("{:.3}", v).len()).max().unwrap_or(0);
                    col_widths[i] = col_widths[i].max(max_data_len);
                } else if let Some(vec) = col.data.downcast_ref::<Vec<i32>>() {
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

        let num_rows = self.columns.values().next().map(|col| col.len).unwrap_or(0);
        for row in 0..num_rows {
            let index = self.index.get(row).map(String::as_str).unwrap_or("");
            write!(f, "| {:^index_width$} |", index, index_width = index_width)?;

            for (label, width) in col_labels.iter().zip(&col_widths) {
                let col = &self.columns[*label];
                if let Some(dtype) = col.dtype {
                    match dtype {
                        "bool" => {
                            let vec = col.data.downcast_ref::<Vec<bool>>().unwrap();
                            let value = vec.get(row).map(|v| v.to_string()).unwrap_or_else(|| "".to_string());
                            write!(f, " {:^width$} |", value, width = width)?;
                        }
                        "i64" => {
                            let vec = col.data.downcast_ref::<Vec<i64>>().unwrap();
                            let value = vec.get(row).map(|v| v.to_string()).unwrap_or_else(|| "".to_string());
                            write!(f, " {:^width$} |", value, width = width)?;
                        }
                        "f64" => {
                            let vec = col.data.downcast_ref::<Vec<f64>>().unwrap();
                            let value = vec.get(row).map(|v| format!("{:.3}", v)).unwrap_or_else(|| "".to_string());
                            write!(f, " {:^width$} |", value, width = width)?;
                        }
                        "alloc::string::String" | "String" => {
                            let vec = col.data.downcast_ref::<Vec<String>>().unwrap();
                            let value = vec.get(row).map(String::as_str).unwrap_or("");
                            write!(f, " {:^width$} |", value, width = width)?;
                        }
                        _ => println!("Unknown dtype '{}'", dtype),
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}