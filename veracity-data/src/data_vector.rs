use std::any::Any;

use ndarray::Array1;

use crate::enums::error_types::DataLoaderError;

pub struct DataVector {
    pub label: Option<String>,
    pub data: Box<dyn Any + Send + Sync>,
    pub len: usize,
    pub dtype: Option<&'static str>
}

impl DataVector {
    pub fn new() -> Self {
        DataVector { 
            label: None, 
            data: Box::new(0),
            len: 0,
            dtype: None
        }
    }

    pub fn from_vec<T: Clone + Send + Sync + 'static>(vec: Vec<T>) -> Result<Self, DataLoaderError> {
        let len: usize = vec.len();
        let dtype: Option<&'static str> = Some(std::any::type_name::<T>());
        Ok(DataVector {
            label: None,
            data: Box::new(vec),
            len,
            dtype
        })
    }

    pub fn to_vec<T: Clone + Send + Sync + 'static>(&self) -> Result<Vec<T>, DataLoaderError> {
        self.data.downcast_ref::<Vec<T>>().ok_or(
            DataLoaderError::GenericError("Failed to cast data to Vec<T>".to_string())
        ).cloned()
    }

    pub fn from_ndarray<T: Clone + Send + Sync + 'static>(arr: Array1<T>) -> Result<Self, DataLoaderError> {
        let dtype: Option<&'static str> = Some(std::any::type_name::<T>());
        Ok(DataVector {
            label: None,
            data: Box::new(arr.to_vec()),
            len: arr.len(),
            dtype
        })
    }

    pub fn to_ndarray<T: Clone + Send + Sync + 'static>(&self) -> Result<Array1<T>, DataLoaderError> {
        let vec: Vec<T> = self.to_vec()?;
        Ok(Array1::from_vec(vec))
    }

    pub fn add_label(&mut self, label: impl AsRef<str>) {
        self.label = Some(label.as_ref().to_string());
    }

    pub fn iter<T: Clone + Send + Sync + 'static>(&self) -> Result<impl Iterator<Item = &T>, DataLoaderError> {
        self.data
            .downcast_ref::<Vec<T>>()
            .map(|vec| vec.iter())
            .ok_or(DataLoaderError::GenericError(
                "Failed to downcast to Vec<T> in iter".to_string(),
            ))
    }

    pub fn iter_mut<T: 'static>(&mut self) -> Result<impl Iterator<Item = &mut T>, DataLoaderError> {
        self.data
            .downcast_mut::<Vec<T>>()
            .map(|vec: &mut Vec<T>| vec.iter_mut())
            .ok_or(DataLoaderError::GenericError(
                "Failed to downcast to Vec<T> for iter_mut".to_string(),
            ))
    }

    pub fn get<T: Clone + Send + Sync + 'static>(&self, index: usize) -> Result<Option<&T>, DataLoaderError> {
        let vec: &Vec<T> = self
            .data
            .downcast_ref::<Vec<T>>()
            .ok_or(DataLoaderError::GenericError(
                "Failed to downcast to Vec<T> in get".to_string(),
            ))?;
        Ok(vec.get(index))
    }

    pub fn get_mut<T: Clone + Send + Sync + 'static>(&mut self, index: usize) -> Result<Option<&mut T>, DataLoaderError> {
        let vec: &mut Vec<T> = self
            .data
            .downcast_mut::<Vec<T>>()
            .ok_or(DataLoaderError::GenericError(
                "Failed to downcast to Vec<T> in get_mut".to_string(),
            ))?;
        Ok(vec.get_mut(index))
    }

    
}

impl Clone for DataVector {
    fn clone(&self) -> Self {
        let label = self.label.clone();
        let dtype = self.dtype;

        let len = self.len;

        let data = if let Some(data_ref) = self.data.downcast_ref::<Vec<i64>>() {
            Box::new(data_ref.clone()) as Box<dyn Any + Send + Sync>
        } else if let Some(data_ref) = self.data.downcast_ref::<Vec<f64>>() {
            Box::new(data_ref.clone()) as Box<dyn Any + Send + Sync>
        } else if let Some(data_ref) = self.data.downcast_ref::<Vec<bool>>() {
            Box::new(data_ref.clone()) as Box<dyn Any + Send + Sync>
        } else if let Some(data_ref) = self.data.downcast_ref::<Vec<String>>() {
            Box::new(data_ref.clone()) as Box<dyn Any + Send + Sync>
        } else {
            unimplemented!("Clone not implemented for this data type")
        };

        DataVector {
            label,
            data,
            len,
            dtype,
        }
    }
}

impl std::fmt::Debug for DataVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataVector {{ label: {:?}, len: {} }}", self.label, self.len)
    }
}

impl std::fmt::Display for DataVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(if let Some(dtype) = self.dtype {
            print!("{}: ", self.label.clone().unwrap_or(" ".to_string()));
            match dtype {
                "bool" => {
                    let vec = self.data.downcast_ref::<Vec<bool>>().unwrap();
                    writeln!(f, "{:?}", vec)?;
                }
                "i64" => {
                    let vec = self.data.downcast_ref::<Vec<i64>>().unwrap();
                    writeln!(f, "{:?}", vec)?;
                }
                "f64" => {
                    let vec = self.data.downcast_ref::<Vec<f64>>().unwrap();
                    writeln!(f, "{:?}", vec)?;
                }
                "alloc::string::String" | "String" => {
                    let vec = self.data.downcast_ref::<Vec<String>>().unwrap();
                    writeln!(f, "{:?}", vec)?;
                }
                _ => writeln!(f, "Unknown dtype '{}'", dtype)?
            }
        })
    }
}