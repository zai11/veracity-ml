use std::collections::BTreeMap;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Dimension;
use num_traits::Num;
use veracity_data::data_matrix::DataMatrix;
use veracity_data::data_vector::DataVector;
use veracity_types::errors::VeracityError;

use super::settings_base::SettingsBase;

pub trait ClassifierBase<D: Dimension, U> {
    fn fit(&mut self, x: &DataMatrix, y: &DataVector<U>) -> Result<(), VeracityError>;

    fn predict(&self, x: &DataMatrix) -> Result<DataVector<U>, VeracityError>;

    fn predict_proba(&self, x: &DataMatrix) -> Result<DataMatrix, VeracityError>;

    fn score(&self, x: &DataMatrix, y: &DataVector<U>) -> Result<f64, VeracityError>;

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), VeracityError>;
}