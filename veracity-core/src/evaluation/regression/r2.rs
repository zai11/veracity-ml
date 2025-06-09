use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;



pub fn _r2<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static,
{
    assert_eq!(
        y_pred.len(),
        y_actual.len(),
        "Predicted and actual arrays must be the same length"
    );

    if y_pred.is_empty() {
        return Ok(f64::NAN);
    }

    let y_actual_f64: Vec<f64> = y_actual.iter().map(|&v| v.into()).collect();
    let y_pred_f64: Vec<f64> = y_pred.iter().map(|&v| v.into()).collect();

    let mean_actual: f64 = y_actual_f64.iter().sum::<f64>() / y_actual_f64.len() as f64;

    let ss_tot: f64 = y_actual_f64
        .iter()
        .map(|&val| (val - mean_actual).powi(2))
        .sum();

    let ss_res: f64 = y_actual_f64
        .iter()
        .zip(y_pred_f64.iter())
        .map(|(&act, &pred)| (act - pred).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return Ok(0.0);
    }

    Ok(1.0 - (ss_res / ss_tot))
}

pub fn r2<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _r2(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}