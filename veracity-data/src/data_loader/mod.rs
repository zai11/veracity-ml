use async_trait::async_trait;

use crate::{data_matrix::DataMatrix, enums::error_types::DataLoaderError};

pub mod data_loader_settings;
pub mod data_types;
pub mod csv_loader;


#[async_trait]
pub trait DataLoader {
    async fn load_from<'a>(&'a self, path: &'a str) -> Result<DataMatrix, DataLoaderError>;
}