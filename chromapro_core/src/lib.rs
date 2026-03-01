mod db;
mod errors;

use parking_lot::Mutex;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::db::Storage;
use crate::errors::ChromaProError;

fn to_py_err(err: ChromaProError) -> PyErr {
    match err {
        ChromaProError::InvalidInput(msg) => PyValueError::new_err(msg),
        ChromaProError::Closed => PyRuntimeError::new_err("database closed"),
        ChromaProError::RocksDb(e) => PyIOError::new_err(e.to_string()),
        ChromaProError::Json(e) => PyValueError::new_err(e.to_string()),
        ChromaProError::Utf8(e) => PyValueError::new_err(e.to_string()),
    }
}

fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(v) => v.into_py(py),
        serde_json::Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                i.into_py(py)
            } else if let Some(u) = v.as_u64() {
                u.into_py(py)
            } else if let Some(f) = v.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(v) => v.into_py(py),
        serde_json::Value::Array(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                let _ = list.append(json_value_to_py(py, item));
            }
            list.into_py(py)
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in obj {
                let _ = dict.set_item(k, json_value_to_py(py, v));
            }
            dict.into_py(py)
        }
    }
}

#[pyclass]
pub struct ChromaProStorage {
    inner: Mutex<Option<Storage>>,
}

#[pymethods]
impl ChromaProStorage {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let storage = Storage::open(&path).map_err(to_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(storage)),
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        Ok(slf)
    }

    fn __exit__(
        &self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    fn close(&self) -> PyResult<()> {
        let mut guard = self.inner.lock();
        if let Some(storage) = guard.as_ref() {
            storage.flush().map_err(to_py_err)?;
        }
        *guard = None;
        Ok(())
    }

    fn ping(&self) -> PyResult<bool> {
        let guard = self.inner.lock();
        if guard.is_none() {
            return Err(to_py_err(ChromaProError::Closed));
        }
        Ok(true)
    }

    fn put_collection(
        &self,
        collection_id: String,
        name: String,
        metadata: String,
    ) -> PyResult<()> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .put_collection(&collection_id, &name, &metadata)
            .map_err(to_py_err)
    }

    fn list_collections(&self, py: Python<'_>) -> PyResult<PyObject> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        let rows = storage.list_collections().map_err(to_py_err)?;
        let out = PyList::empty_bound(py);
        for row in rows {
            let dict = PyDict::new_bound(py);
            let id = row.get("id").cloned().unwrap_or_default();
            let name = row.get("name").cloned().unwrap_or_default();
            let metadata_raw = row
                .get("metadata")
                .cloned()
                .unwrap_or_else(|| "{}".to_string());

            dict.set_item("id", id)?;
            dict.set_item("name", name)?;

            let metadata_json = serde_json::from_str::<serde_json::Value>(&metadata_raw)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
            dict.set_item("metadata", json_value_to_py(py, &metadata_json))?;
            out.append(dict)?;
        }
        Ok(out.into_py(py))
    }

    fn batch_insert(
        &self,
        collection_id: String,
        ids: Vec<String>,
        embeddings: Vec<Vec<u8>>,
        documents: Vec<String>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .batch_insert(&collection_id, ids, embeddings, documents, metadatas)
            .map_err(to_py_err)
    }

    fn delete_embeddings(&self, collection_id: String, ids: Vec<String>) -> PyResult<usize> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .delete_embeddings(&collection_id, ids)
            .map_err(to_py_err)
    }

    fn get_embeddings(
        &self,
        collection_id: String,
        ids: Vec<String>,
    ) -> PyResult<Vec<(Vec<u8>, String, String)>> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .get_embeddings(&collection_id, ids)
            .map_err(to_py_err)
    }

    fn list_embedding_ids(&self, collection_id: String) -> PyResult<Vec<String>> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .list_embedding_ids(&collection_id)
            .map_err(to_py_err)
    }

    fn delete_collection_data(&self, collection_id: String) -> PyResult<()> {
        let guard = self.inner.lock();
        let storage = guard
            .as_ref()
            .ok_or_else(|| to_py_err(ChromaProError::Closed))?;

        storage
            .delete_collection_data(&collection_id)
            .map_err(to_py_err)
    }
}

#[pymodule]
fn chromapro_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ChromaProStorage>()?;
    Ok(())
}
