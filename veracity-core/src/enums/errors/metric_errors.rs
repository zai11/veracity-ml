use std::fmt;

use veracity_types::errors::VeracityError;


#[derive(Debug)]
pub enum EvaluationMetricError {
    GenericError(String)
}

impl fmt::Display for EvaluationMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationMetricError::GenericError(e) => write!(f, "An error occurred in evaluation metric:\r\n{:#?}", e)
        }
    }
}

impl From<EvaluationMetricError> for VeracityError {
    fn from(err: EvaluationMetricError) -> Self {
        VeracityError::EvaluationMetric(format!("{:?}", err))
    }
}

impl std::error::Error for EvaluationMetricError {}