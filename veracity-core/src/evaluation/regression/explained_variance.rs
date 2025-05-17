use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _explained_variance<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static,
{
    assert_eq!(
        y_pred.len(),
        y_actual.len(),
        "Predicted and actual arrays must be the same length"
    );

    let n: usize = y_actual.len();
    if n == 0 {
        return f64::NAN;
    }

    let actual_vals: Vec<f64> = y_actual.iter().map(|y: &U| (*y).into()).collect();
    let pred_vals: Vec<f64> = y_pred.iter().map(|y: &T| (*y).into()).collect();

    let mean_actual: f64 = actual_vals.iter().sum::<f64>() / n as f64;

    let var_actual: f64 = actual_vals
        .iter()
        .map(|y: &f64| (y - mean_actual).powi(2))
        .sum::<f64>()
        / n as f64;

    let var_diff: f64 = actual_vals
        .iter()
        .zip(pred_vals.iter())
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum::<f64>()
        / n as f64;

    if var_actual == 0.0 {
        if var_diff == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - (var_diff / var_actual)
    }
}

pub fn explained_variance<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _explained_variance(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}