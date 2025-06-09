use ndarray::{Array1, Array2, Dimension};
use num_traits::Num;
use veracity_data::{data_matrix::DataMatrix, data_vector::DataVector};
use veracity_types::errors::VeracityError;

use super::settings_base::SettingsBase;



pub trait RegressorBase<D: Dimension> {
    fn fit(&mut self, x: &DataMatrix, y: &DataVector<f64>) -> Result<(), VeracityError>;

    fn predict(&self, x: &DataMatrix) -> Result<DataVector<f64>, VeracityError>;

    fn score(&self, x: &DataMatrix, y: &DataVector<f64>) -> Result<f64, VeracityError>;

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), VeracityError>;
}