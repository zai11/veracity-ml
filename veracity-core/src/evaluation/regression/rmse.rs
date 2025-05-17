use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _rmse<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
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

    let mse: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(&pred, &actual)| {
            let error = pred.into() - actual.into();
            error * error
        })
        .sum::<f64>()
        / y_pred.len() as f64;

    mse.sqrt()
}

pub fn rmse<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _rmse(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}