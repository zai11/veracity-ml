use std::{cmp::Ordering, collections::HashSet, sync::{Arc, Mutex}};

use std::hash::Hash;

use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVector, TDataVectorExt}, enums::error_types::DataLoaderError};

use crate::enums::errors::metric_errors::EvaluationMetricError;


// TODO: Ensure only two classes are allowed. Return Error otherwise.
pub fn auprc<T, U>(proba_pred: &DataMatrix, actual: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Clone + Send + Sync + 'static,
    U: Eq + Hash + Clone + Send + Sync + 'static,
{
    if proba_pred.nrows() != actual.len() {
        return Err(EvaluationMetricError::GenericError(
            "Inputs must be the same length".to_string(),
        ));
    }

    // Check that there are only two unique classes
    let unique_classes: HashSet<U> = actual.iter().map_err(|e| EvaluationMetricError::GenericError(e.to_string()))?.cloned().collect();

    if unique_classes.len() != 2 {
        return Err(EvaluationMetricError::GenericError(format!(
            "AUPRC is only defined for binary classification. Found {} classes.",
            unique_classes.len()
        )));
    }

    // Use the label of the first value as the "positive" class.
    let binding: Vec<U> = actual.to_vec().map_err(|e: DataLoaderError| EvaluationMetricError::GenericError(e.to_string()))?;
    let positive_class: &U = match binding.first() {
        Some(val) => val,
        None => return Ok(0.0),
    };

    // Extract column 1 from proba_pred (e.g., the predicted probability for class 1).
    let column: Arc<Mutex<dyn TDataVector>> = proba_pred
        .column_at(1)
        .ok_or_else(|| EvaluationMetricError::GenericError("Missing predicted probability column".to_string()))?;

    let column_guard = column.lock().map_err(|_| EvaluationMetricError::GenericError("Mutex poisoned".to_string()))?;
    let column_vector = column_guard
        .as_any()
        .downcast_ref::<DataVector<T>>()
        .ok_or_else(|| EvaluationMetricError::GenericError("Invalid column format".to_string()))?;

    let predicted_probs: Vec<f64> = column_vector
        .to_vec().map_err(|e| EvaluationMetricError::GenericError(e.to_string()))?
        .into_iter()
        .map(|val| val.into())
        .collect();

    let actual_labels: Vec<U> = actual.to_vec().map_err(|e| EvaluationMetricError::GenericError(e.to_string()))?;

    let mut paired: Vec<(f64, u8)> = predicted_probs
        .into_iter()
        .zip(actual_labels.into_iter())
        .map(|(score, label)| {
            let binary = if &label == positive_class { 1 } else { 0 };
            (score, binary)
        })
        .collect();

    // Sort by predicted score descending
    paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let total_positives: f64 = paired.iter().filter(|&&(_, y)| y == 1).count() as f64;
    if total_positives == 0.0 {
        return Ok(0.0);
    }

    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;
    let mut precision: Vec<f64> = vec![];
    let mut recall: Vec<f64> = vec![];

    for &(_, label) in &paired {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        precision.push(tp / (tp + fp));
        recall.push(tp / total_positives);
    }

    // Calculate AUPRC using the trapezoidal rule
    let mut auprc: f64 = 0.0;
    for i in 1..recall.len() {
        let delta_recall = recall[i] - recall[i - 1];
        auprc += precision[i] * delta_recall;
    }

    Ok(auprc)
}