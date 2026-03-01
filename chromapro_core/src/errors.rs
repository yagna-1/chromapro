use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChromaProError {
    #[error("database is closed")]
    Closed,
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("rocksdb error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("utf8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}
