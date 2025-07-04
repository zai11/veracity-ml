use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _mape<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
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

    let mut total_error: f64 = 0.0;
    let mut valid_count: i32 = 0;

    for (pred, actual) in y_pred.iter().zip(y_actual.iter()) {
        let pred = (*pred).into();
        let actual = (*actual).into();

        if actual != 0.0 {
            total_error += ((actual - pred).abs()) / actual.abs();
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        f64::NAN
    } else {
        (total_error / valid_count as f64) * 100.0
    }
}

pub fn mape<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _mape(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}