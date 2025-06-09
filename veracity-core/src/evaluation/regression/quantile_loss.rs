use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _quantile_loss<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>, quantile: &f64) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static,
{
    assert_eq!(
        y_pred.len(),
        y_actual.len(),
        "Predicted and actual arrays must be the same length"
    );
    assert!(
        (0.0..=1.0).contains(quantile),
        "Quantile must be between 0 and 1"
    );

    if y_pred.is_empty() {
        return Ok(f64::NAN);
    }

    let total_loss: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(&pred, &actual)| {
            let pred_f = pred.into();
            let actual_f = actual.into();
            let error = actual_f - pred_f;
            if error >= 0.0 {
                quantile * error
            } else {
                (1.0 - quantile) * -error
            }
        })
        .sum();

    Ok(total_loss / y_pred.len() as f64)
}

pub fn quantile_loss<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>, quantile: &f64) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _quantile_loss(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap(), quantile)
}