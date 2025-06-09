
#[derive(Clone, Debug)]
pub enum EvaluationMetrics {
    // Classification Metrics
    Accuracy,
    AUPRC,
    AUROC,
    F1,
    GeometricMean,
    LogLoss,
    LRAP,
    Precision,
    Recall,
    Specificity,

    // Clustering Metrics
    CalinskiHarabaszIndex,
    DaviesBouldinIndex,
    DunnIndex,
    SilhouetteCoefficient,

    // Regression Metrics
    AdjustedR2,
    ExplainedVarience,
    HuberLoss,
    LogCoshLoss,
    MAE,
    MAPE,
    MBD,
    MSE,
    MSLE,
    QuantileLoss,
    R2,
    RMSE,
    SMAPE
}