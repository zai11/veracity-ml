use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _log_loss<T, U>(proba_pred: &Array1<T>, actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(proba_pred.len(), actual.len(), "Arrays must be the same length.");

    let positive_class: &U = match actual.first() {
        Some(val) => val,
        None => return Ok(0.0),
    };

    let epsilon: f64 = 1e-15;
    let mut total_loss: f64 = 0.0;

    for (pred, actual) in proba_pred.iter().zip(actual.iter()) {
        let prob: f64 = pred.clone().into().clamp(epsilon, 1.0 - epsilon);
        let is_positive: bool = actual == positive_class;

        if is_positive {
            total_loss += -prob.ln();
        } else {
            total_loss += -(1.0 - prob).ln();
        }
    }

    Ok(total_loss / proba_pred.len() as f64)
}

pub fn log_loss<T, U>(proba_pred: &DataVector<T>, actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _log_loss(&proba_pred.to_ndarray().unwrap(), &actual.to_ndarray().unwrap())
}