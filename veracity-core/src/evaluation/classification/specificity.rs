use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _specificity<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    if y_actual.len() == 0 {
        return 0.0;
    }

    let positive_class = match y_actual.first() {
        Some(val) => val,
        None => return 0.0,
    };

    let mut true_negatives = 0.0;
    let mut false_positives = 0.0;

    for (pred, actual) in y_pred.iter().zip(y_actual.iter()) {
        if actual != positive_class {
            if pred == actual {
                true_negatives += 1.0;
            } else if pred == positive_class {
                false_positives += 1.0;
            }
        }
    }

    if true_negatives + false_positives == 0.0 {
        0.0
    } else {
        true_negatives / (true_negatives + false_positives)
    }
}

pub fn specificity<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _specificity(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}