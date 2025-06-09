use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;

pub fn _f1<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64,  EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;
    let mut fn_: f64 = 0.0;

    for (pred, actual) in y_pred.iter().zip(y_actual.iter()) {
        if pred == actual {
            tp += 1.0;
        } else {
            fp += 1.0;
            fn_ += 1.0;
        }
    }

    let precision: f64 = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
    let recall: f64 = if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };

    if precision + recall == 0.0 {
        Ok(0.0)
    } else {
        Ok(2.0 * (precision * recall) / (precision + recall))
    }
}

pub fn f1<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    _f1(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}