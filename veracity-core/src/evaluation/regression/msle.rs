use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _msle<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
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

    let mut total_squared_log_error: f64 = 0.0;
    let mut valid_count: i32 = 0;

    for (&pred, &actual) in y_pred.iter().zip(y_actual.iter()) {
        let pred_f = pred.into();
        let actual_f = actual.into();

        if pred_f >= 0.0 && actual_f >= 0.0 {
            let log_pred = (1.0 + pred_f).ln();
            let log_actual = (1.0 + actual_f).ln();
            total_squared_log_error += (log_pred - log_actual).powi(2);
            valid_count += 1;
        } else {
            return Ok(f64::NAN);
        }
    }

    Ok(total_squared_log_error / valid_count as f64)
}

pub fn msle<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _msle(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}