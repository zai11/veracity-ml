use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _rmse<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
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

    let mse: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(&pred, &actual)| {
            let error = pred.into() - actual.into();
            error * error
        })
        .sum::<f64>()
        / y_pred.len() as f64;

    Ok(mse.sqrt())
}

pub fn rmse<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _rmse(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}