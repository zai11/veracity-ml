use std::{any::Any, cmp::Ordering, collections::BTreeMap, sync::{Arc, Mutex}, hash::Hash};

use ndarray::{Array1, ArrayBase, Ix2, OwnedRepr};
use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVector, TDataVectorExt}};
use veracity_types::errors::VeracityError;

use crate::{base::{classifier_base::ClassifierBase, settings_base::SettingsBase}, enums::{distance_metrics::DistanceMetrics, evaluation_metrics::EvaluationMetrics}, evaluation::classification::{accuracy::accuracy, auprc::auprc, f1::f1, geometric_mean::geometric_mean, precision::precision, recall::recall, specificity::specificity}, utility::distance::{find_distance_cosine, find_distance_euclidean, find_distance_manhatten, find_distance_minkowski, find_distance_nan_euclidean}};

use super::k_neighbors_weights::KNeighborsWeights;

#[derive(Clone)]
pub struct KNeighborsClassifierSettings {
    pub k_neighbors: usize,
    pub weights: KNeighborsWeights,
    pub p: i64,
    pub distance_metric: DistanceMetrics,
    pub score_metric: EvaluationMetrics,
    pub epsilon: f64
}

impl SettingsBase for KNeighborsClassifierSettings {}

impl Default for KNeighborsClassifierSettings {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            weights: KNeighborsWeights::Uniform,
            p: 2,
            distance_metric: DistanceMetrics::Euclidean,
            score_metric: EvaluationMetrics::Accuracy,
            epsilon: f64::EPSILON
        }
    }
}

pub struct KNeighborsClassifier<U> {
    x: Option<DataMatrix>,
    y: Option<DataVector<U>>,
    settings: KNeighborsClassifierSettings
}

impl<U> KNeighborsClassifier<U> {
    pub fn new() -> Self {
        KNeighborsClassifier {
            x: None,
            y: None,
            settings: KNeighborsClassifierSettings::default()
        }
    }

    pub fn check_for_nan(&self, x: &ArrayBase<OwnedRepr<f32>, Ix2>) -> bool {
        x.iter().any(|&val| val.is_nan())
    }
}

impl<U> ClassifierBase<Ix2, U> for KNeighborsClassifier<U> 
where
    U: std::fmt::Debug + Eq + Hash + Ord + Clone + Send + Sync + ToString + 'static
{
    fn fit(&mut self, x: &DataMatrix, y: &DataVector<U>) -> Result<(), VeracityError> {
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
        Ok(())
    }

    fn predict(&self, x: &DataMatrix) -> Result<DataVector<U>, VeracityError> {
        let x_train = self.x.as_ref().ok_or_else(|| VeracityError::GenericError("Classifier has not been fitted.".into()))?;
        let y_train = self.y.as_ref().ok_or_else(|| VeracityError::GenericError("Classifier has not been fitted.".into()))?;

        let feature_names: Vec<_> = x_train.columns.keys().cloned().collect();

        // Extract rows as Vec<Vec<f64>>
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
        let y_vec: Vec<U> = y_train.iter()?.cloned().collect();

        let mut predictions: Vec<U> = Vec::with_capacity(x_test_rows.len());

        for test_row in x_test_rows {
            let mut distances: Vec<(f64, &U)> = x_train_rows
                .iter()
                .zip(y_vec.iter())
                .map(|(train_row, label)| {
                    let distance = match self.settings.distance_metric {
                        DistanceMetrics::Euclidean => find_distance_euclidean(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Cosine => find_distance_cosine(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Manhatten => find_distance_manhatten(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Minkowski => find_distance_minkowski(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view(), &self.settings.p),
                        DistanceMetrics::NanEuclidean => find_distance_nan_euclidean(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                    };
                    (distance, label)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let k_neighbors = &distances[..self.settings.k_neighbors.min(distances.len())];

            let mut vote_counts: BTreeMap<U, f64> = BTreeMap::new();
            for (distance, label) in k_neighbors {
                match self.settings.weights {
                    KNeighborsWeights::Uniform => {
                        *vote_counts.entry((*label).clone()).or_insert(0.0) += 1.0;
                    },
                    KNeighborsWeights::Distance => {
                        let weight = 1.0 / (distance + self.settings.epsilon);
                        *vote_counts.entry((*label).clone()).or_insert(0.0) += weight;
                    }
                }
            }

            let predicted = vote_counts
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(label, _)| label)
                .ok_or_else(|| VeracityError::GenericError("Unable to determine prediction.".to_string()))?;

            predictions.push(predicted);
        }

        let mut result = DataVector::from_vec(predictions)?;
        result.add_label("predictions");
        Ok(result)
    }

    fn predict_proba(&self, x: &DataMatrix) -> Result<DataMatrix, VeracityError> {
        let x_train = self.x.as_ref().ok_or_else(|| VeracityError::GenericError("Classifier has not been fitted.".into()))?;
        let y_train = self.y.as_ref().ok_or_else(|| VeracityError::GenericError("Classifier has not been fitted.".into()))?;

        let feature_names: Vec<_> = x_train.columns.keys().cloned().collect();

        // Extract rows
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
                .map(|i| columns.iter().map(|col| col[i]).collect())
                .collect())
        };

        let x_train_rows = extract_rows(x_train)?;
        let x_test_rows = extract_rows(x)?;
        let y_vec: Vec<U> = y_train.iter()?.cloned().collect();

        let mut unique_labels = y_vec.clone();
        unique_labels.sort();
        unique_labels.dedup();
        let n_classes = unique_labels.len();
        let n_samples = x_test_rows.len();

        // Prepare a vector of vectors for each class's probabilities
        let mut class_columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_classes];

        for test_row in x_test_rows.iter() {
            let mut distances: Vec<(f64, &U)> = x_train_rows
                .iter()
                .zip(y_vec.iter())
                .map(|(train_row, label)| {
                    let distance = match self.settings.distance_metric {
                        DistanceMetrics::Euclidean => find_distance_euclidean(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Cosine => find_distance_cosine(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Manhatten => find_distance_manhatten(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                        DistanceMetrics::Minkowski => find_distance_minkowski(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view(), &self.settings.p),
                        DistanceMetrics::NanEuclidean => find_distance_nan_euclidean(&Array1::from(test_row.clone()).view(), &Array1::from(train_row.clone()).view()),
                    };
                    (distance, label)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let k_neighbors = &distances[..self.settings.k_neighbors.min(distances.len())];

            let mut class_votes: BTreeMap<U, f64> = BTreeMap::new();
            for (dist, label) in k_neighbors {
                let weight = match self.settings.weights {
                    KNeighborsWeights::Uniform => 1.0,
                    KNeighborsWeights::Distance => 1.0 / (dist + self.settings.epsilon),
                };
                *class_votes.entry((*label).clone()).or_insert(0.0) += weight;
            }

            let total_weight: f64 = class_votes.values().sum();
            for (j, class) in unique_labels.iter().enumerate() {
                let vote = class_votes.get(class).copied().unwrap_or(0.0);
                class_columns[j].push(vote / total_weight);
            }
        }

        // Convert class_columns into DataMatrix
        let mut class_vectors: Vec<Arc<Mutex<DataVector<f64>>>> = Vec::with_capacity(n_classes);
        for (i, col) in class_columns.into_iter().enumerate() {
            let mut dv = DataVector::from_vec(col)?;
            dv.add_label(&unique_labels[i].to_string());
            class_vectors.push(Arc::new(Mutex::new(dv)));
        }

        let class_vectors: Vec<Arc<Mutex<dyn TDataVector>>> = class_vectors
            .into_iter()
            .map(|dv| dv as Arc<Mutex<dyn TDataVector>>)
            .collect();

        Ok(DataMatrix::from_vec(class_vectors)?)
    }

    fn score(&self, x: &DataMatrix, y: &DataVector<U>) -> Result<f64, VeracityError> {
        match self.settings.score_metric {
            EvaluationMetrics::Accuracy => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(accuracy(&y_pred, y)?)
            },
            EvaluationMetrics::AUPRC => {
                let proba_predict: DataMatrix = self.predict_proba(x)?;
                Ok(auprc::<f64, U>(&proba_predict, y)?)
            },
            EvaluationMetrics::AUROC => {
                Err(VeracityError::NotImplemented)
                /*let proba_predict: DataMatrix = self.predict_proba(x)?;
                Ok(auroc(&proba_predict, y)?)*/
            },
            EvaluationMetrics::F1 => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(f1(&y_pred, y)?)
            },
            EvaluationMetrics::GeometricMean => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(geometric_mean(&y_pred, y)?)
            },
            EvaluationMetrics::LogLoss => {
                Err(VeracityError::NotImplemented)
                /*let proba_predict: DataMatrix = self.predict_proba(x)?;
                Ok(log_loss(&proba_predict, y)?)*/
            },
            EvaluationMetrics::LRAP => {
                Err(VeracityError::NotImplemented)
                /*let y_pred: DataVector<U> = self.predict(x)?;
                Ok(lrap(&y_pred, y)?)*/
            },
            EvaluationMetrics::Precision => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(precision(&y_pred, y)?)
            },
            EvaluationMetrics::Recall => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(recall(&y_pred, y)?)
            },
            EvaluationMetrics::Specificity => {
                let y_pred: DataVector<U> = self.predict(x)?;
                Ok(specificity(&y_pred, y)?)
            },
            EvaluationMetrics::CalinskiHarabaszIndex => {
                Err(VeracityError::Classifier("Unable to use Calinski-Harabasz Index evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::DaviesBouldinIndex => {
                Err(VeracityError::Classifier("Unable to use Davies-Bouldin Index evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::DunnIndex => {
                Err(VeracityError::Classifier("Unable to use Dunn Index evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::SilhouetteCoefficient => {
                Err(VeracityError::Classifier("Unable to use Silhouette Coefficient evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::AdjustedR2 => {
                Err(VeracityError::Classifier("Unable to use Adjusted R^2 evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::ExplainedVarience => {
                Err(VeracityError::Classifier("Unable to use Explained Variance evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::HuberLoss => {
                Err(VeracityError::Classifier("Unable to use Huber Loss evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::LogCoshLoss => {
                Err(VeracityError::Classifier("Unable to use Log-Cosh Loss evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::MAE => {
                Err(VeracityError::Classifier("Unable to use MAE evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::MAPE => {
                Err(VeracityError::Classifier("Unable to use MAPE evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::MBD => {
                Err(VeracityError::Classifier("Unable to use MBD evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::MSE => {
                Err(VeracityError::Classifier("Unable to use MSE evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::MSLE => {
                Err(VeracityError::Classifier("Unable to use MSLE evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::QuantileLoss => {
                Err(VeracityError::Classifier("Unable to use Quantile Loss evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::R2 => {
                Err(VeracityError::Classifier("Unable to use R^2 evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::RMSE => {
                Err(VeracityError::Classifier("Unable to use RMSE evaluation metric on a classifier.".to_string()))
            },
            EvaluationMetrics::SMAPE => {
                Err(VeracityError::Classifier("Unable to use SMAPE evaluation metric on a classifier.".to_string()))
            }
        }
    }

    fn add_settings<S: SettingsBase + 'static>(&mut self, settings: S) -> Result<(), VeracityError> {
        let any: &dyn Any = &settings as &dyn Any;

        if let Some(settings) = any.downcast_ref::<KNeighborsClassifierSettings>() {
            self.settings = settings.clone();
            Ok(())
        } else {
            Err(VeracityError::Classifier("Invalid settings type passed to KNeighborsClassifier".to_string()))
        }
    }
}