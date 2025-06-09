use std::{any::Any, fmt::{self, Display}};

use itertools::Itertools;
use ndarray::Array1;

use crate::enums::error_types::DataLoaderError;

pub trait TDataVector
{
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
    fn dtype(&self) -> &'static str;
    fn add_label(&mut self, label: &str);
    fn get_label(&self) -> Option<String>;
    fn get_data(&self) -> &dyn Any;
    fn append(&mut self, value: &dyn Any) -> Result<(), DataLoaderError>;
}

pub trait TDataVectorExt<T> : TDataVector
where 
    T: Clone + Send + Sync + 'static
{
    fn new() -> Self where Self: Sized;
    fn from_vec(vec: Vec<T>) -> Result<Self, DataLoaderError> where Self: Sized;
    fn to_vec(&self) -> Result<Vec<T>, DataLoaderError>;
    fn from_ndarray(arr: Array1<T>) -> Result<Self, DataLoaderError> where Self: Sized;
    fn to_ndarray(&self) -> Result<Array1<T>, DataLoaderError>;
    fn iter(&self) -> Result<impl Iterator<Item = &T>, DataLoaderError>;
    fn iter_mut(&mut self) -> Result<impl Iterator<Item = &mut T>, DataLoaderError>;
    fn get(&self, index: usize) -> Result<Option<&T>, DataLoaderError>;
    fn get_mut(&mut self, index: usize) -> Result<Option<&mut T>, DataLoaderError>;
}

pub struct DataVector<T> {
    label: Option<String>,
    data: Vec<T>
}

impl<T> TDataVector for DataVector<T>
where 
    T: Clone + Send + Sync + 'static
{
    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn add_label(&mut self, label: &str) {
        self.label = Some(label.to_string());
    }

    fn get_label(&self) -> Option<String> {
        self.label.clone()
    }

    fn get_data(&self) -> &dyn Any {
        &self.data
    }

    fn append(&mut self, value: &dyn Any) -> Result<(), DataLoaderError> {
        let value = value
            .downcast_ref::<T>()
            .ok_or_else(|| DataLoaderError::GenericError("Error converting passed value to T".to_string()))?;
        self.data.push(value.clone());
        Ok(())
    }
}

impl<T> TDataVectorExt<T> for DataVector<T>
where
    T: Clone + Send + Sync + 'static
{
    fn new() -> Self {
        DataVector { 
            label: None, 
            data: Vec::new()
        }
    }

    fn from_vec(vec: Vec<T>) -> Result<Self, DataLoaderError> {
        Ok(DataVector {
            label: None,
            data: vec,
        })
    }

    fn to_vec(&self) -> Result<Vec<T>, DataLoaderError> {
        Ok(self.data.to_owned())
    }

    fn from_ndarray(arr: Array1<T>) -> Result<Self, DataLoaderError> {
        Ok(DataVector {
            label: None,
            data: arr.to_vec(),
        })
    }

    fn to_ndarray(&self) -> Result<Array1<T>, DataLoaderError> {
        Ok(Array1::from_vec(self.data.clone()))
    }

    fn iter(&self) -> Result<impl Iterator<Item = &T>, DataLoaderError> {
        Ok(self.data.iter())
    }

    fn iter_mut(&mut self) -> Result<impl Iterator<Item = &mut T>, DataLoaderError> {
        Ok(self.data.iter_mut())
    }

    fn get(&self, index: usize) -> Result<Option<&T>, DataLoaderError> {
        Ok(self.data.get(index))
    }

    fn get_mut(&mut self, index: usize) -> Result<Option<&mut T>, DataLoaderError> {
        Ok(self.data.get_mut(index))
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for DataVector<T> {
    fn clone(&self) -> Self {
        DataVector {
            label: self.label.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T: Clone + Send + Sync + 'static> std::fmt::Debug for DataVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataVector {{ label: {:?}, len: {} }}", self.label, self.data.len())
    }
}

impl<T: Clone + Display + Send + Sync + 'static> Display for DataVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label: String = self.label.clone().unwrap_or_else(|| " ".to_string());
        let joined_data = self.data.iter().map(|x: &T| x.to_string()).join(" ");
        write!(f, "{}: {}", label, joined_data)
    }
}

pub fn downcast_ref<T: 'static>(v: &dyn TDataVector) -> Option<&DataVector<T>> {
    v.as_any().downcast_ref::<DataVector<T>>()
}