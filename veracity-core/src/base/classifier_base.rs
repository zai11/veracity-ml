use std::collections::BTreeMap;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Dimension;
use ndarray::OwnedRepr;
use num_traits::Num;
use veracity_data::data_matrix::DataMatrix;
use veracity_data::data_vector::DataVector;
use veracity_types::errors::VeracityError;

use super::settings_base::SettingsBase;

pub trait ClassifierBase<T: Num + Copy, D: Dimension, U> {
    fn _fit(&mut self, x: &Array2<T>, y: &Array1<U>) -> Result<(), VeracityError>;

    fn fit(&mut self, x: &DataMatrix, y: &DataVector) -> Result<(), VeracityError>;

    fn _predict(&self, x: &Array2<T>) -> Result<Array1<U>, VeracityError>;

    fn predict(&self, x: &DataMatrix) -> Result<DataVector, VeracityError>;

    fn _predict_proba(&self, x: &Array2<T>) -> Result<Vec<BTreeMap<U, f64>>, VeracityError>;

    fn predict_proba(&self, x: &DataMatrix) -> Result<Vec<DataMatrix>, VeracityError>;

    fn _score(&self, x: &Array2<T>, y: &Array1<U>) -> Result<f64, VeracityError>;

    fn score(&self, x: &DataMatrix, y: &DataVector) -> Result<f64, VeracityError>;

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), VeracityError>;
}