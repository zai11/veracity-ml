use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _geometric_mean<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    let positive_class: &U = match y_actual.first() {
        Some(val) => val,
        None => return Ok(0.0),
    };

    let mut tp: f64 = 0.0;
    let mut tn: f64 = 0.0;
    let mut fp: f64 = 0.0;
    let mut fn_: f64 = 0.0;

    for (pred, actual) in y_pred.iter().zip(y_actual.iter()) {
        let is_pred_pos: bool = pred == positive_class;
        let is_actual_pos: bool = actual == positive_class;

        match (is_pred_pos, is_actual_pos) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            (false, false) => tn += 1.0,
        }
    }

    let recall: f64 = if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
    let specificity: f64 = if tn + fp == 0.0 { 0.0 } else { tn / (tn + fp) };

    Ok((recall * specificity).sqrt())
}

pub fn geometric_mean<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _geometric_mean(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}