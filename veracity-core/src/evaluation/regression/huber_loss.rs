use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _huber_loss<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>, delta: &f64) -> f64
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
        return f64::NAN;
    }

    let losses: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(y_hat, y)| {
            let y_hat = (*y_hat).into();
            let y = (*y).into();
            let error = y - y_hat;
            let abs_error = error.abs();

            if abs_error <= *delta {
                0.5 * error.powi(2)
            } else {
                delta * (abs_error - 0.5 * delta)
            }
        })
        .sum();

    losses / y_pred.len() as f64
}

pub fn huber_loss<T, U>(y_pred: &DataVector, y_actual: &DataVector, delta: &f64) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _huber_loss(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap(), delta)
}