use ndarray::Array1;
use veracity_data::data_vector::{DataVector, TDataVectorExt};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _specificity<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    if y_actual.len() == 0 {
        return Ok(0.0);
    }

    let positive_class: &U = match y_actual.first() {
        Some(val) => val,
        None => return Ok(0.0),
    };

    let mut true_negatives: f64 = 0.0;
    let mut false_positives: f64 = 0.0;

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
        Ok(0.0)
    } else {
        Ok(true_negatives / (true_negatives + false_positives))
    }
}

pub fn specificity<T, U>(y_pred: &DataVector<T>, y_actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _specificity(&y_pred.to_ndarray().unwrap(), &y_actual.to_ndarray().unwrap())
}