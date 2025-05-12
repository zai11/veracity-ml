use std::fmt;

use veracity_types::errors::VeracityError;

#[derive(Debug)]
pub enum DataLoaderError {
    DuplicateIndex,
    IndexRowCountMismatch,
    RowCountMismatch,
    ColumnCountMismatch(String),
    DuplicateLabel,
    NDArrayConversion(String),
    HeterogeneousDataTypes,
    HeterogeneousColumnLengths,
    NoData,
    IndexError(usize),
    FileRead(String),
    GenericError(String)
}

impl fmt::Display for DataLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataLoaderError::DuplicateIndex => write!(f, "DataFrame's index must not have duplicate values"),
            DataLoaderError::IndexRowCountMismatch => write!(f, "DataFrame's index must have the same number of rows as the columns"),
            DataLoaderError::RowCountMismatch => write!(f, "All columns must have the same number of rows."),
            DataLoaderError::ColumnCountMismatch(e) => write!(f, "{:#?}", e),
            DataLoaderError::DuplicateLabel => write!(f, "DataFrame columns may not have duplicate labels."),
            DataLoaderError::NDArrayConversion(e) => write!(f, "An error occurred converting between ndarray and data vector:\r\n{:#?}", e),
            DataLoaderError::HeterogeneousDataTypes => write!(f, "Unable to convert DataMatrix with heterogeneous data types to ndarray Array2."),
            DataLoaderError::HeterogeneousColumnLengths => write!(f, "All columns in a DataMatrix must be the same length."),
            DataLoaderError::NoData => write!(f, "The DataMatrix contains no data"),
            DataLoaderError::IndexError(index) => write!(f, "No element was found at index: {}", index),
            DataLoaderError::FileRead(e) => write!(f, "An error occurred reading from file:\r\n{:#?}", e),
            DataLoaderError::GenericError(e) => write!(f, "An error occurred in DataLoader:\r\n{:#?}", e)
        }
    }
}

impl From<DataLoaderError> for VeracityError {
    fn from(err: DataLoaderError) -> Self {
        VeracityError::DataLoader(format!("{:?}", err))
    }
}

impl std::error::Error for DataLoaderError {}