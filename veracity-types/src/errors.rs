use std::fmt;

#[derive(Debug)]
pub enum VeracityError {
    DataLoader(String),
    Classifier(String),
    Regressor(String),
    NotImplemented,
    GenericError(String)
}

impl fmt::Display for VeracityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VeracityError::DataLoader(e) => write!(f, "An error occurred while loading data:\r\n{:#?}", e),
            VeracityError::Classifier(e) => write!(f, "An error occurred in a classification model:\r\n{:#?}", e),
            VeracityError::Regressor(e) => write!(f, "An error called in a regression model:\r\n{:#?}", e),
            VeracityError::NotImplemented => write!(f, "The called function is not implemented"),
            VeracityError::GenericError(e) => write!(f, "{:#?}", e),
        }
    }
}