use std::{any::Any, cmp::Ordering, collections::BTreeMap, iter::Sum};

use ndarray::{Array1, Array2, ArrayBase, Axis, Ix2, OwnedRepr};
use num_traits::{Float, Num, ToPrimitive};
use veracity_data::{data_matrix::DataMatrix, data_vector::DataVector};
use veracity_types::errors::VeracityError;

use crate::{base::{classifier_base::Classifier, settings_base::ClassifierSettingsBase}, enums::distance_metrics::DistanceMetrics, utility::distance::{find_distance_cosine, find_distance_euclidean, find_distance_manhatten, find_distance_minkowski, find_distance_nan_euclidean}};

#[derive(Clone, Debug)]
pub enum KNeighborsWeights {
    Uniform,
    Distance
}

#[derive(Clone)]
pub struct KNeighborsClassifierSettings {
    pub k_neighbors: usize,
    pub weights: KNeighborsWeights,
    pub p: i64,
    pub metric: DistanceMetrics,
    pub epsilon: f64
}

impl ClassifierSettingsBase for KNeighborsClassifierSettings {}

impl Default for KNeighborsClassifierSettings {
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

pub struct KNeighborsClassifier<T: Num + Copy, U> {
    x: Option<Array2<T>>,
    y: Option<Array1<U>>,
    settings: KNeighborsClassifierSettings
}

impl<T, U> KNeighborsClassifier<T, U> where T: Num + Copy {
    pub fn new() -> Self {
        KNeighborsClassifier {
            x: None,
            y: None,
            settings: KNeighborsClassifierSettings::default()
        }
    }
}

impl<U> KNeighborsClassifier<f32, U> {
    pub fn check_for_nan(&self, x: &ArrayBase<OwnedRepr<f32>, Ix2>) -> bool {
        x.iter().any(|&val| val.is_nan())
    }
}

impl<T: Float + Sum + Send + Sync + 'static, U: std::fmt::Debug + Clone + Ord + Send + Sync + 'static> Classifier<T, Ix2, U> for KNeighborsClassifier<T, U> where T: Num + ToPrimitive + Copy, U: Eq + Clone {
    fn _fit(&mut self, x: ArrayBase<OwnedRepr<T>, Ix2>, y: Array1<U>) -> Result<(), VeracityError> {
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn fit(&mut self, x: &DataMatrix, y: &DataVector) -> Result<(), VeracityError> {
        let x = x.to_ndarray()?;
        let y = y.to_ndarray()?;
        self._fit(x, y)
    }

    fn _predict(&self, x_test: ArrayBase<OwnedRepr<T>, Ix2>) -> Result<Array1<U>, VeracityError> {
        let x_train = self.x.as_ref().expect("KNeighborsClassifier must be trained on a feature matrix");
        let y_train = self.y.as_ref().expect("KNeighborsClassifier must be trained on a label vector");

        let mut predictions: Vec<U> = Vec::with_capacity(x_test.nrows());
        
        for test_point in x_test.axis_iter(Axis(0)) {
            let mut distances = x_train
            .axis_iter(Axis(0))
            .zip(y_train.iter())
            .map(|(train_point, label)| {
                let distance = match self.settings.metric {
                    DistanceMetrics::Cosine => find_distance_cosine::<T>(&test_point, &train_point),
                    DistanceMetrics::Euclidean => find_distance_euclidean::<T>(&test_point, &train_point),
                    DistanceMetrics::Manhatten =>find_distance_manhatten::<T>(&test_point, &train_point),
                    DistanceMetrics::Minkowski => find_distance_minkowski::<T>(&test_point, &train_point, self.settings.p),
                    DistanceMetrics::NanEuclidean => find_distance_nan_euclidean::<T>(&test_point, &train_point)
                };
                (distance, label)
            })
            .collect::<Vec<_>>();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let k_neighbors = &distances[0..self.settings.k_neighbors];

            //println!("Neighbours:");
            //for neighbour in k_neighbors {
            //    println!("{:#?}, {:#?}", neighbour.0, neighbour.1);
            //}

            let mut vote_counts: BTreeMap<U, f64> = BTreeMap::new();
            for (distance, label) in k_neighbors {
                match self.settings.weights {
                    KNeighborsWeights::Uniform => {
                        *vote_counts.entry((*label).clone()).or_insert(0.0) += 1.0;
                    },
                    KNeighborsWeights::Distance => {
                        let weight: f64 = 1.0 / (distance + self.settings.epsilon);
                        *vote_counts.entry((*label).clone()).or_insert(0.0) += weight;
                    }
                };
            }

            let predicted_value = vote_counts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
            .to_owned();

            predictions.push(predicted_value)
        }

        Ok(Array1::from(predictions))

    }

    fn predict(&self, x_test: &DataMatrix) -> Result<DataVector, VeracityError> {
        let result: ArrayBase<OwnedRepr<U>, ndarray::Dim<[usize; 1]>> = self._predict(x_test.to_ndarray()?)?;
        let mut data_vector = DataVector::from_ndarray(result)?;
        data_vector.add_label("predictions");
        Ok(data_vector)
    }

    fn _predict_proba(&self, x: ArrayBase<OwnedRepr<T>, Ix2>) -> Result<Vec<BTreeMap<U, f64>>, VeracityError> {
        let x_train = self.x.as_ref().ok_or(VeracityError::NotImplemented)?;
        let y_train = self.y.as_ref().ok_or(VeracityError::NotImplemented)?;

        let mut probabilities = Vec::with_capacity(x.len_of(Axis(0)));

        for sample in x.outer_iter() {
            // Compute distances to all training points
            let mut distances: Vec<(usize, T)> = x_train
                .outer_iter()
                .enumerate()
                .map(|(i, train_sample)| {
                    let dist = sample
                        .iter()
                        .zip(train_sample.iter())
                        .map(|(a, b)| (*a - *b).powi(2))
                        .sum::<T>()
                        .sqrt();
                    (i, dist)
                })
                .collect();

            // Sort by distance and select k nearest neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let k_nearest = distances.iter().take(self.settings.k_neighbors).map(|(i, _)| &y_train[*i]);

            // Count class frequencies
            let mut class_counts: BTreeMap<U, usize> = BTreeMap::new();
            for label in k_nearest {
                *class_counts.entry(label.clone()).or_insert(0) += 1;
            }

            //println!("Class Counts: {:#?}", class_counts);

            // Convert to probabilities
            let mut probs: BTreeMap<U, f64> = BTreeMap::new();
            for (label, count) in class_counts {
                probs.insert(label, count as f64 / self.settings.k_neighbors as f64);
            }

            probabilities.push(probs);
        }

        Ok(probabilities)
    }

    fn predict_proba(&self, x: &DataMatrix) -> Result<Vec<DataMatrix>, VeracityError> {
        let result:Vec<BTreeMap<U, f64>>  = self._predict_proba(x.to_ndarray()?)?;
        Ok(result
            .into_iter()
            .map(|probs| {
                let class_labels: Vec<U> = probs.keys().cloned().collect();
                let probabilities: Vec<f64> = probs.values().cloned().collect();

                let mut dm = DataMatrix::new();
                _ = dm.add_column(class_labels, Some("class"));
                _ = dm.add_column(probabilities, Some("probabilities"));
                dm
            })
            .collect())
    }

    fn _score(&self, x: ArrayBase<OwnedRepr<T>, Ix2>, y: Array1<U>) -> Result<f64, VeracityError> {
        let y_pred = self._predict(x)?;

        if y_pred.len() != y.len() {
            return Err(VeracityError::GenericError("Dimensions Don't Match".to_string()));
        }

        let correct = y.iter()
            .zip(y_pred.iter())
            .filter(|(true_label, pred_label)| true_label == pred_label)
            .count();

        let accuracy = correct as f64 / y.len() as f64;
        Ok(accuracy)
    }

    fn score(&self, x: &DataMatrix, y: &DataVector) -> Result<f64, VeracityError> {
        Ok(self._score(x.to_ndarray()?, y.to_ndarray()?)?)
        /*let predicted = self.predict(x)?;
    
        macro_rules! try_score {
            ($t:ty) => {{
                if let (Some(pred), Some(actual)) = (
                    predicted.data.downcast_ref::<Vec<$t>>(),
                    y.data.downcast_ref::<Vec<$t>>(),
                ) {
                    let correct = pred.iter().zip(actual.iter()).filter(|(a, b)| a == b).count();
                    return Ok(correct as f64 / pred.len() as f64);
                }
            }};
        }
    
        try_score!(bool);
        try_score!(i64);
        try_score!(f64);
        try_score!(String);
    
        Err(VeracityError::GenericError("Unsupported data type".to_string()))*/
    }

    fn add_settings<S: ClassifierSettingsBase + 'static>(&mut self, settings: S) -> Result<(), VeracityError> {
        let any: &dyn Any = &settings as &dyn Any;

        if let Some(knn_settings) = any.downcast_ref::<KNeighborsClassifierSettings>() {
            self.settings = knn_settings.clone();
            Ok(())
        } else {
            Err(VeracityError::Classifier("Invalid settings type passed to KNeighborsClassifier".to_string()))
        }
    }
}