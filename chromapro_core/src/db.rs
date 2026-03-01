use std::collections::HashMap;

use rocksdb::{ColumnFamilyDescriptor, Direction, IteratorMode, Options, WriteBatch, DB};

use crate::errors::ChromaProError;

pub const CF_METADATA: &str = "metadata";
pub const CF_EMBEDDINGS: &str = "embeddings";
pub const CF_DOCUMENTS: &str = "documents";

pub struct Storage {
    pub db: DB,
}

impl Storage {
    pub fn open(path: &str) -> Result<Self, ChromaProError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_write_buffer_size(64 * 1024 * 1024);
        opts.set_max_open_files(1000);
        opts.set_use_fsync(true);
        opts.set_wal_bytes_per_sync(1024 * 1024);

        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_METADATA, Options::default()),
            ColumnFamilyDescriptor::new(CF_EMBEDDINGS, Options::default()),
            ColumnFamilyDescriptor::new(CF_DOCUMENTS, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)?;
        Ok(Self { db })
    }

    pub fn flush(&self) -> Result<(), ChromaProError> {
        self.db.flush()?;
        Ok(())
    }

    pub fn put_collection(
        &self,
        collection_id: &str,
        name: &str,
        metadata: &str,
    ) -> Result<(), ChromaProError> {
        let cf = self
            .db
            .cf_handle(CF_METADATA)
            .ok_or_else(|| ChromaProError::InvalidInput("metadata CF not found".to_string()))?;

        let key = format!("collection::{collection_id}");
        let value = serde_json::json!({
            "id": collection_id,
            "name": name,
            "metadata": metadata,
        })
        .to_string();

        self.db.put_cf(&cf, key.as_bytes(), value.as_bytes())?;
        self.db.flush_cf(&cf)?;
        Ok(())
    }

    pub fn list_collections(&self) -> Result<Vec<HashMap<String, String>>, ChromaProError> {
        let cf = self
            .db
            .cf_handle(CF_METADATA)
            .ok_or_else(|| ChromaProError::InvalidInput("metadata CF not found".to_string()))?;

        let prefix = b"collection::";
        let iter = self
            .db
            .iterator_cf(&cf, IteratorMode::From(prefix, Direction::Forward));

        let mut out = Vec::new();
        for item in iter {
            let (key, value) = item?;
            if !key.starts_with(prefix) {
                break;
            }

            let parsed: serde_json::Value = serde_json::from_slice(&value)?;
            let mut row = HashMap::new();
            row.insert(
                "id".to_string(),
                parsed
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            );
            row.insert(
                "name".to_string(),
                parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            );
            row.insert(
                "metadata".to_string(),
                parsed
                    .get("metadata")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string(),
            );
            out.push(row);
        }

        Ok(out)
    }

    pub fn batch_insert(
        &self,
        collection_id: &str,
        ids: Vec<String>,
        embeddings: Vec<Vec<u8>>,
        documents: Vec<String>,
        metadatas: Vec<String>,
    ) -> Result<usize, ChromaProError> {
        if ids.len() != embeddings.len()
            || ids.len() != documents.len()
            || ids.len() != metadatas.len()
        {
            return Err(ChromaProError::InvalidInput(
                "ids/embeddings/documents/metadatas length mismatch".to_string(),
            ));
        }

        let cf_emb = self
            .db
            .cf_handle(CF_EMBEDDINGS)
            .ok_or_else(|| ChromaProError::InvalidInput("embeddings CF not found".to_string()))?;
        let cf_doc = self
            .db
            .cf_handle(CF_DOCUMENTS)
            .ok_or_else(|| ChromaProError::InvalidInput("documents CF not found".to_string()))?;

        let mut batch = WriteBatch::default();
        for i in 0..ids.len() {
            let emb_key = format!("{collection_id}::{}", ids[i]);
            batch.put_cf(&cf_emb, emb_key.as_bytes(), &embeddings[i]);

            let doc_key = format!("{collection_id}::{}", ids[i]);
            let doc_value = serde_json::json!({
                "document": documents[i],
                "metadata": metadatas[i],
            })
            .to_string();
            batch.put_cf(&cf_doc, doc_key.as_bytes(), doc_value.as_bytes());
        }

        self.db.write(batch)?;
        self.db.flush()?;
        Ok(ids.len())
    }

    pub fn delete_embeddings(
        &self,
        collection_id: &str,
        ids: Vec<String>,
    ) -> Result<usize, ChromaProError> {
        let cf_emb = self
            .db
            .cf_handle(CF_EMBEDDINGS)
            .ok_or_else(|| ChromaProError::InvalidInput("embeddings CF not found".to_string()))?;
        let cf_doc = self
            .db
            .cf_handle(CF_DOCUMENTS)
            .ok_or_else(|| ChromaProError::InvalidInput("documents CF not found".to_string()))?;

        let mut batch = WriteBatch::default();
        for id in &ids {
            let emb_key = format!("{collection_id}::{id}");
            let doc_key = format!("{collection_id}::{id}");
            batch.delete_cf(&cf_emb, emb_key.as_bytes());
            batch.delete_cf(&cf_doc, doc_key.as_bytes());
        }

        self.db.write(batch)?;
        self.db.flush()?;
        Ok(ids.len())
    }

    pub fn get_embeddings(
        &self,
        collection_id: &str,
        ids: Vec<String>,
    ) -> Result<Vec<(Vec<u8>, String, String)>, ChromaProError> {
        let cf_emb = self
            .db
            .cf_handle(CF_EMBEDDINGS)
            .ok_or_else(|| ChromaProError::InvalidInput("embeddings CF not found".to_string()))?;
        let cf_doc = self
            .db
            .cf_handle(CF_DOCUMENTS)
            .ok_or_else(|| ChromaProError::InvalidInput("documents CF not found".to_string()))?;

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let emb_key = format!("{collection_id}::{id}");
            let doc_key = format!("{collection_id}::{id}");

            let embedding = self
                .db
                .get_cf(&cf_emb, emb_key.as_bytes())?
                .map(|v| v.to_vec())
                .unwrap_or_default();

            let doc_data = self
                .db
                .get_cf(&cf_doc, doc_key.as_bytes())?
                .map(|v| v.to_vec())
                .unwrap_or_default();

            let doc_json: serde_json::Value = if doc_data.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_slice(&doc_data).unwrap_or(serde_json::Value::Null)
            };

            let document = doc_json
                .get("document")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let metadata = doc_json
                .get("metadata")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();

            results.push((embedding, document, metadata));
        }

        Ok(results)
    }

    pub fn list_embedding_ids(&self, collection_id: &str) -> Result<Vec<String>, ChromaProError> {
        let cf_emb = self
            .db
            .cf_handle(CF_EMBEDDINGS)
            .ok_or_else(|| ChromaProError::InvalidInput("embeddings CF not found".to_string()))?;

        let prefix = format!("{collection_id}::");
        let iter = self.db.iterator_cf(
            &cf_emb,
            IteratorMode::From(prefix.as_bytes(), Direction::Forward),
        );

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            let key_str = String::from_utf8(key.to_vec())?;
            if let Some((_, id)) = key_str.split_once("::") {
                ids.push(id.to_string());
            }
        }

        Ok(ids)
    }

    pub fn delete_collection_data(&self, collection_id: &str) -> Result<(), ChromaProError> {
        let cf_meta = self
            .db
            .cf_handle(CF_METADATA)
            .ok_or_else(|| ChromaProError::InvalidInput("metadata CF not found".to_string()))?;
        let cf_emb = self
            .db
            .cf_handle(CF_EMBEDDINGS)
            .ok_or_else(|| ChromaProError::InvalidInput("embeddings CF not found".to_string()))?;
        let cf_doc = self
            .db
            .cf_handle(CF_DOCUMENTS)
            .ok_or_else(|| ChromaProError::InvalidInput("documents CF not found".to_string()))?;

        let mut batch = WriteBatch::default();
        let meta_key = format!("collection::{collection_id}");
        batch.delete_cf(&cf_meta, meta_key.as_bytes());

        let prefix = format!("{collection_id}::");

        let emb_iter = self.db.iterator_cf(
            &cf_emb,
            IteratorMode::From(prefix.as_bytes(), Direction::Forward),
        );
        for item in emb_iter {
            let (key, _) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            batch.delete_cf(&cf_emb, key);
        }

        let doc_iter = self.db.iterator_cf(
            &cf_doc,
            IteratorMode::From(prefix.as_bytes(), Direction::Forward),
        );
        for item in doc_iter {
            let (key, _) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            batch.delete_cf(&cf_doc, key);
        }

        self.db.write(batch)?;
        self.db.flush()?;
        Ok(())
    }
}
