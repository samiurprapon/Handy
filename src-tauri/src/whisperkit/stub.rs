//! Stub engine for platforms (or builds) without the WhisperKit Swift
//! package. Same API as the real engine; everything reports unsupported.

use super::WhisperKitError;
use std::path::Path;

pub struct WhisperKitEngine;

impl WhisperKitEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn supported() -> bool {
        false
    }

    pub fn version() -> String {
        "unavailable".into()
    }

    pub fn load_model(&self, _model_dir: &Path) -> Result<(), WhisperKitError> {
        Err(WhisperKitError::NotSupported)
    }

    pub fn unload_model(&self) {}

    pub fn is_model_loaded(&self) -> bool {
        false
    }

    pub fn transcribe(
        &self,
        _samples: &[f32],
        _language: Option<&str>,
    ) -> Result<String, WhisperKitError> {
        Err(WhisperKitError::NotSupported)
    }

    pub fn cancel(&self) {}
}

impl Default for WhisperKitEngine {
    fn default() -> Self {
        Self::new()
    }
}
