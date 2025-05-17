use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _log_cosh_loss<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
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

    let total_loss: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(y_hat, y)| {
            let diff = (*y).into() - (*y_hat).into();
            (diff.cosh()).ln()
        })
        .sum();

    total_loss / y_pred.len() as f64
}

pub fn log_cosh_loss<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _log_cosh_loss(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}