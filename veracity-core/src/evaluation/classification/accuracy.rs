use veracity_data::data_vector::{DataVector, TDataVectorExt};
use crate::enums::errors::metric_errors::EvaluationMetricError;

pub fn accuracy<U: PartialEq>(y_true: &DataVector<U>, y_pred: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where 
    U: Clone + Send + Sync + 'static
{
    let true_vals: Vec<U> = y_true.to_vec().map_err(|e| EvaluationMetricError::GenericError(e.to_string()))?;
    let pred_vals: Vec<U> = y_pred.to_vec().map_err(|e| EvaluationMetricError::GenericError(e.to_string()))?;

    if true_vals.len() != pred_vals.len() {
        return Err(EvaluationMetricError::GenericError("Vectors have different lengths".to_string()));
    }

    let correct: usize = true_vals.iter()
        .zip(pred_vals.iter())
        .filter(|(a, b)| a == b)
        .count();

    Ok(correct as f64 / true_vals.len() as f64)
}