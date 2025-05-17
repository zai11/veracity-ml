use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _smape<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
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

    let mut smape_sum: f64 = 0.0;
    let n: f64 = y_pred.len() as f64;

    for (&pred, &actual) in y_pred.iter().zip(y_actual.iter()) {
        let pred_f = pred.into();
        let actual_f = actual.into();
        let numerator = (pred_f - actual_f).abs();
        let denominator = (pred_f.abs() + actual_f.abs()) / 2.0;

        let term: f64 = if denominator == 0.0 { 0.0 } else { numerator / denominator };
        smape_sum += term;
    }

    (smape_sum / n) * 100.0
}

pub fn smape<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _smape(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}