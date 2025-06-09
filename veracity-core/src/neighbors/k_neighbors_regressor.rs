use core::f64;
use std::any::Any;

use ndarray::{Array1, Ix2};
use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVector, TDataVectorExt}};
use veracity_types::errors::VeracityError;

use crate::{base::{regressor_base::RegressorBase, settings_base::SettingsBase}, enums::distance_metrics::DistanceMetrics, utility::distance::{find_distance_cosine, find_distance_euclidean, find_distance_manhatten, find_distance_minkowski, find_distance_nan_euclidean}};

use super::k_neighbors_weights::KNeighborsWeights;

#[derive(Clone)]
pub struct KNeighborsRegressorSettings {
    pub k_neighbors: usize,
    pub weights: KNeighborsWeights,
    pub p: i64,
    pub metric: DistanceMetrics,
    pub epsilon: f64
}

impl SettingsBase for KNeighborsRegressorSettings {}

impl Default for KNeighborsRegressorSettings {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            weights: KNeighborsWeights::Uniform,
            p: 2,
            metric: DistanceMetrics::Euclidean,
            epsilon: f64::EPSILON
        }
    }
}

pub struct KNeighborsRegressor {
    x: Option<DataMatrix>,
    y: Option<DataVector<f64>>,
    settings: KNeighborsRegressorSettings
}

impl KNeighborsRegressor {
    pub fn new() -> Self {
        KNeighborsRegressor {
            x: None,
            y: None,
            settings: KNeighborsRegressorSettings::default()
        }
    }
}


impl RegressorBase<Ix2> for KNeighborsRegressor {
    fn fit(&mut self, x: &DataMatrix, y: &DataVector<f64>) -> Result<(), VeracityError> {
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
        Ok(())
    }

    fn predict(&self, x: &DataMatrix) -> Result<DataVector<f64>, VeracityError> {
        let x_train = self.x.as_ref().ok_or_else(|| VeracityError::GenericError("Regressor has not been fitted.".into()))?;
        let y_train = self.y.as_ref().ok_or_else(|| VeracityError::GenericError("Regressor has not been fitted.".into()))?;

        let feature_names: Vec<_> = x_train.columns.keys().cloned().collect();

        // Helper function to extract row vectors
        let extract_rows = |data: &DataMatrix| -> Result<Vec<Vec<f64>>, VeracityError> {
            let columns: Vec<Vec<f64>> = feature_names
                .iter()
                .map(|name| {
                    let col = data.columns.get(name).ok_or_else(|| VeracityError::GenericError("Invalid data format.".into()))?;
                    let locked = col.lock().unwrap();
                    let vec = locked
                        .as_any()
                        .downcast_ref::<DataVector<f64>>()
                        .ok_or_else(|| VeracityError::GenericError("Invalid data format.".into()))?
                        .to_vec()?;
                    Ok::<_, VeracityError>(vec)
                })
                .collect::<Result<_, _>>()?;

            let n_rows = data.nrows();
            Ok((0..n_rows)
                .map(|i| columns.iter().map(|col| col[i].clone()).collect())
                .collect())
        };

        let x_train_rows = extract_rows(x_train)?;
        let x_test_rows = extract_rows(x)?;
        let y_vec: Vec<f64> = y_train.iter()?.cloned().collect();

        let mut predictions = Vec::with_capacity(x_test_rows.len());

        for row in x_test_rows {
            let mut distances: Vec<(f64, f64)> = x_train_rows.iter()
                .zip(y_vec.iter())
                .map(|(train_row, &label)| {
                    let distance = match self.settings.metric {
                        DistanceMetrics::Euclidean => find_distance_euclidean(&Array1::from(row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Cosine => find_distance_cosine(&Array1::from(row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Manhatten => find_distance_manhatten(&Array1::from(row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Minkowski => find_distance_minkowski(&Array1::from(row.clone()).view(), &Array1::from(train_row.clone()).view(), &self.settings.p),
                        DistanceMetrics::NanEuclidean => find_distance_nan_euclidean(&Array1::from(row.clone()).view(), &Array1::from(train_row.clone()).view()),
                    };
                    (distance, label)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let neighbors = &distances[..self.settings.k_neighbors.min(distances.len())];

            let prediction = match self.settings.weights {
                KNeighborsWeights::Uniform => {
                    let sum: f64 = neighbors.iter().map(|(_, val)| *val).sum();
                    sum / neighbors.len() as f64
                }
                KNeighborsWeights::Distance => {
                    let mut weighted_sum = 0.0;
                    let mut total_weight = 0.0;
                    for (dist, val) in neighbors {
                        let weight = 1.0 / (dist + self.settings.epsilon);
                        weighted_sum += weight * val;
                        total_weight += weight;
                    }
                    weighted_sum / total_weight
                }
            };

            predictions.push(prediction);
        }

        let mut output = DataVector::from_vec(predictions)?;
        output.add_label("predictions");
        Ok(output)
    }

    fn score(&self, x: &DataMatrix, y: &DataVector<f64>) -> Result<f64, VeracityError> {
        let y_pred = self.predict(x)?;
    
        if y.len() != y_pred.len() {
            return Err(VeracityError::GenericError("Dimensions Don't Match".to_string()));
        }
    
        let y_true = y.iter()?.copied().collect::<Vec<f64>>();
        let y_pred_vec = y_pred.iter()?.copied().collect::<Vec<f64>>();
    
        let mean_y = y_true.iter().sum::<f64>() / y_true.len() as f64;
    
        let ss_tot = y_true.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
        let ss_res = y_true.iter().zip(y_pred_vec.iter()).map(|(yi, ypi)| (yi - ypi).powi(2)).sum::<f64>();
    
        Ok(1.0 - (ss_res / ss_tot))
    }

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), veracity_types::errors::VeracityError> {
        let any: &dyn Any = &settings as &dyn Any;

        if let Some(settings) = any.downcast_ref::<KNeighborsRegressorSettings>() {
            self.settings = settings.clone();
            Ok(())
        } else {
            Err(VeracityError::Classifier("Invalid settings type passed to KNeighborsRegressor".to_string()))
        }
    }
}
