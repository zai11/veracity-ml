use std::{any::Any, iter::Sum};

use ndarray::{Array1, Array2, Ix2};
use num_traits::{Float, Num, ToPrimitive};
use veracity_data::{data_matrix::DataMatrix, data_vector::DataVector};
use veracity_types::errors::VeracityError;

use crate::{base::{regressor_base::RegressorBase, settings_base::SettingsBase}, enums::distance_metrics::DistanceMetrics, utility::distance::{find_distance_cosine, find_distance_euclidean, find_distance_manhatten, find_distance_minkowski, find_distance_nan_euclidean}};

use super::k_neighbors_weights::KNeighborsWeights;

#[derive(Clone)]
pub struct KNeighborsRegressorSettings {
    pub k_neighbors: i64,
    pub weights: KNeighborsWeights,
    pub p: i64,
    pub metric: DistanceMetrics
}

impl SettingsBase for KNeighborsRegressorSettings {}

impl Default for KNeighborsRegressorSettings {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            weights: KNeighborsWeights::Uniform,
            p: 2,
            metric: DistanceMetrics::Euclidean,
        }
    }
}

pub struct KNeighborsRegressor<T: Num + Copy, U> {
    x: Option<Array2<T>>,
    y: Option<Array1<U>>,
    settings: KNeighborsRegressorSettings
}

impl<T, U> KNeighborsRegressor<T, U> where T: Num + Copy {
    pub fn new() -> Self {
        KNeighborsRegressor {
            x: None,
            y: None,
            settings: KNeighborsRegressorSettings::default()
        }
    }
}


impl<T: Copy + Float + Sync + Send + ToPrimitive + 'static, U: Clone + Sync + Send + Float + Sum + 'static> RegressorBase<T, Ix2, U> for KNeighborsRegressor<T, U> {
    fn _fit(&mut self, x: &Array2<T>, y: &Array1<U>) -> Result<(), VeracityError> {
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
        Ok(())
    }

    fn fit(&mut self, x: &DataMatrix, y: &DataVector) -> Result<(), VeracityError> {
        let x: Array2<T> = x.to_ndarray()?;
        let y: Array1<U> = y.to_ndarray()?;
        self._fit(&x, &y)
    }

    fn _predict(&self, x: &Array2<T>) -> Result<Array1<U>, VeracityError> {
        let x_train: &Array2<T> = self.x.as_ref().expect("KNeighborsClassifier must be trained on a feature matrix");
        let y_train: &Array1<U> = self.y.as_ref().expect("KNeighborsClassifier must be trained on a label vector");

        let mut predictions = Vec::with_capacity(x.nrows());

    for row in x.outer_iter() {
        let mut distances: Vec<(_, U)> = x_train.outer_iter()
            .zip(y_train.iter())
            .map(|(train_row, y_val)| {
                let distance: f64 = match self.settings.metric {
                    DistanceMetrics::Cosine => find_distance_cosine::<T>(&row, &train_row),
                    DistanceMetrics::Euclidean => find_distance_euclidean::<T>(&row, &train_row),
                    DistanceMetrics::Manhatten =>find_distance_manhatten::<T>(&row, &train_row),
                    DistanceMetrics::Minkowski => find_distance_minkowski::<T>(&row, &train_row, &self.settings.p),
                    DistanceMetrics::NanEuclidean => find_distance_nan_euclidean::<T>(&row, &train_row)
                };
                (distance, *y_val)
            })
            .collect::<Vec<_>>();

        distances.sort_by(|a: &(f64, U), b: &(f64, U)| a.0.partial_cmp(&b.0).unwrap());
        let neighbors: &[(f64, U)] = &distances[..self.settings.k_neighbors as usize];

        let prediction = match self.settings.weights {
            KNeighborsWeights::Uniform => {
                let sum = neighbors.iter().map(|&(_, val)| val).sum::<U>();
                sum / U::from(self.settings.k_neighbors).unwrap()
            }
            KNeighborsWeights::Distance => {
                let mut weighted_sum: U = U::zero();
                let mut weight_total: T = T::zero();

                for &(dist, val) in neighbors {
                    let weight: T = T::from(1.0 / dist).unwrap_or_else(T::zero);

                    weighted_sum = weighted_sum + val * U::from(weight).unwrap();
                    weight_total = weight_total + weight;
                }

                weighted_sum / U::from(weight_total).unwrap()
            }
        };

        predictions.push(prediction);
    }

    Ok(Array1::from(predictions))
    }

    fn predict(&self, x: &DataMatrix) -> Result<DataVector, VeracityError> {
        let result: Array1<U> = self._predict(&x.to_ndarray()?)?;
        let mut data_vector: DataVector = DataVector::from_ndarray(result)?;
        data_vector.add_label("predictions");
        Ok(data_vector)
    }

    fn _score(&self, x: &Array2<T>, y: &Array1<U>) -> Result<f64, VeracityError> {
        let y_pred = self._predict(x)?;

        if y.len() != y_pred.len() {
            return Err(VeracityError::GenericError("Dimensions Don't Match".to_string()));
        }

        let mean_y = y.iter().cloned().sum::<U>() / U::from(y.len()).unwrap();
        let ss_tot: f64 = y.iter().map(|yi| {
            let diff = *yi - mean_y;
            diff.to_f64().unwrap().powi(2)
        }).sum();

        let ss_res: f64 = y.iter().zip(y_pred.iter()).map(|(yi, ypi)| {
            let diff = *yi - *ypi;
            diff.to_f64().unwrap().powi(2)
        }).sum();

        Ok(1.0 - (ss_res / ss_tot))
    }

    fn score(&self, x: &veracity_data::data_matrix::DataMatrix, y: &veracity_data::data_vector::DataVector) -> Result<f64, veracity_types::errors::VeracityError> {
        Ok(self._score(&x.to_ndarray()?, &y.to_ndarray()?)?)
    }

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), veracity_types::errors::VeracityError> {
        let any: &dyn Any = &settings as &dyn Any;

        if let Some(settings) = any.downcast_ref::<KNeighborsRegressorSettings>() {
            self.settings = settings.clone();
            Ok(())
        } else {
            Err(VeracityError::Classifier("Invalid settings type passed to KNeighborsClassifier".to_string()))
        }
    }
}
