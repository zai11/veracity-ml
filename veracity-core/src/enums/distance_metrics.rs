#[derive(Clone, Debug)]
pub enum DistanceMetrics {
    Cosine,
    Euclidean,
    Manhatten,
    Minkowski,
    NanEuclidean
}