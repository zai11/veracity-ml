use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _quantile_loss<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>, quantile: &f64) -> f64
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
        return f64::NAN;
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

    total_loss / y_pred.len() as f64
}

pub fn quantile_loss<T, U>(y_pred: &DataVector, y_actual: &DataVector, quantile: &f64) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _quantile_loss(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap(), quantile)
}