//! Safe wrapper around the WhisperKit FFI. macOS arm64 only.
//!
//! One engine lives for the process; the model inside it loads and unloads
//! on demand. The mutex guards only the handle; the Swift actor serializes
//! the work. Every call but `cancel` blocks — invoke from worker threads
//! only (the transcription paths that reach this run on background threads).

use super::ffi;
use super::WhisperKitError;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Mutex;

pub struct WhisperKitEngine {
    /// The Swift handle. Created lazily on first use so a process that never
    /// selects a WhisperKit model never pays for the Swift runtime startup.
    /// `None` after [`Self::unload_handle`] (only on drop).
    handle: Mutex<Option<*mut std::ffi::c_void>>,
}

// SAFETY: the handle is created and destroyed behind `handle`'s mutex, the
// Swift object behind it is an actor safe to call from any thread, and Rust
// never touches that object directly.
unsafe impl Send for WhisperKitEngine {}
unsafe impl Sync for WhisperKitEngine {}

/// Read a result's strings out and free it. Exactly one of text/error is set;
/// an error string maps to `Err` with the Swift message verbatim.
///
/// # Safety
/// `raw` must be a pointer returned by a fallible `whisperkit_*` call and not
/// yet freed.
unsafe fn take_result(
    raw: *mut ffi::WhisperKitResult,
    cancelled_marker: &str,
) -> Result<String, WhisperKitError> {
    if raw.is_null() {
        return Err(WhisperKitError::TranscribeFailed(
            "WhisperKit bridge returned a null result".into(),
        ));
    }
    let text = (*raw).text;
    let error = (*raw).error;
    let out = if !error.is_null() {
        let message = CStr::from_ptr(error).to_string_lossy().into_owned();
        if message == cancelled_marker {
            Err(WhisperKitError::Cancelled)
        } else {
            Err(WhisperKitError::TranscribeFailed(message))
        }
    } else if !text.is_null() {
        Ok(CStr::from_ptr(text).to_string_lossy().into_owned())
    } else {
        Err(WhisperKitError::TranscribeFailed(
            "WhisperKit bridge returned an empty result".into(),
        ))
    };
    ffi::whisperkit_free_result(raw);
    out
}

impl WhisperKitEngine {
    pub fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }

    /// Whether this build can run WhisperKit at all (platform + OS version).
    pub fn supported() -> bool {
        // SAFETY: takes no arguments and only reads the OS version.
        unsafe { ffi::whisperkit_supported() == 1 }
    }

    /// The pinned WhisperKit version, for logs and the model manager.
    pub fn version() -> String {
        // SAFETY: returns a static C string that lives for the process.
        unsafe {
            let raw = ffi::whisperkit_bridge_version();
            if raw.is_null() {
                return "unknown".into();
            }
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }

    /// The live handle, creating it if needed.
    fn handle(&self) -> Result<*mut std::ffi::c_void, WhisperKitError> {
        let mut guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            // SAFETY: returns a fresh retained handle stored exactly once.
            let handle = unsafe { ffi::whisperkit_create() };
            if handle.is_null() {
                return Err(WhisperKitError::TranscribeFailed(
                    "WhisperKit bridge could not allocate its handle".into(),
                ));
            }
            *guard = Some(handle);
        }
        Ok(guard.unwrap())
    }

    /// Load the WhisperKit model at `model_dir`. Blocks for as long as Core
    /// ML takes (the first load on a machine compiles for the Neural Engine).
    pub fn load_model(&self, model_dir: &Path) -> Result<(), WhisperKitError> {
        if !Self::supported() {
            return Err(WhisperKitError::NotSupported);
        }
        let handle = self.handle()?;
        let path = CString::new(model_dir.to_string_lossy().as_bytes().to_vec())
            .map_err(|_| WhisperKitError::LoadFailed("model path contains a NUL byte".into()))?;
        // SAFETY: `handle` is live (destroyed only on drop); `path` outlives
        // the blocking call.
        let raw = unsafe { ffi::whisperkit_load_model(handle, path.as_ptr()) };
        // SAFETY: fresh result pointer, freed exactly once.
        match unsafe { take_result(raw, "") } {
            Ok(_) => Ok(()),
            Err(WhisperKitError::TranscribeFailed(m)) => Err(WhisperKitError::LoadFailed(m)),
            Err(e) => Err(e),
        }
    }

    /// Unload the current model. Idempotent.
    pub fn unload_model(&self) {
        let guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(handle) = *guard {
            // SAFETY: `handle` is live (destroyed only on drop).
            let raw = unsafe { ffi::whisperkit_unload_model(handle) };
            // SAFETY: fresh result pointer, freed exactly once.
            drop(unsafe { take_result(raw, "") });
        }
    }

    /// Whether a model is currently loaded.
    pub fn is_model_loaded(&self) -> bool {
        let guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        match *guard {
            // SAFETY: `handle` is live (destroyed only on drop).
            Some(handle) => unsafe { ffi::whisperkit_is_model_loaded(handle) == 1 },
            None => false,
        }
    }

    /// Transcribe `samples` (16 kHz mono f32) with `language` forced (e.g.
    /// `Some("bn")`). Blocks until decoding finishes.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<String, WhisperKitError> {
        if !Self::supported() {
            return Err(WhisperKitError::NotSupported);
        }
        if !self.is_model_loaded() {
            return Err(WhisperKitError::NotLoaded);
        }
        if samples.is_empty() {
            return Err(WhisperKitError::InvalidAudio("empty buffer".into()));
        }
        let handle = self.handle()?;
        let language = match language.map(CString::new) {
            Some(Ok(s)) => Some(s),
            Some(Err(_)) => {
                return Err(WhisperKitError::InvalidAudio(
                    "language contains a NUL byte".into(),
                ))
            }
            None => None,
        };
        // SAFETY: `handle` is live (destroyed only on drop); `samples` and
        // `language` outlive the blocking call; the result is freed once.
        let raw = unsafe {
            ffi::whisperkit_transcribe(
                handle,
                samples.as_ptr(),
                samples.len(),
                language.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
            )
        };
        // SAFETY: fresh result pointer, freed exactly once. "cancelled" is
        // the stable marker the Swift side reports for cancellations.
        match unsafe { take_result(raw, "cancelled") } {
            Ok(text) => Ok(text),
            Err(WhisperKitError::TranscribeFailed(m)) => Err(WhisperKitError::TranscribeFailed(m)),
            Err(e) => Err(e),
        }
    }

    /// Request cancellation of the in-flight transcription. Returns
    /// immediately; the blocked [`Self::transcribe`] observes it and fails
    /// with [`WhisperKitError::Cancelled`] (Cooperative: WhisperKit's decode
    /// cannot be preempted mid-window, so a cancel racing a nearly-finished
    /// decode still surfaces after the decode completes).
    pub fn cancel(&self) {
        let guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(handle) = *guard {
            // SAFETY: `handle` is live; the call never blocks.
            unsafe { ffi::whisperkit_cancel(handle) };
        }
    }
}

impl Default for WhisperKitEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WhisperKitEngine {
    fn drop(&mut self) {
        if let Some(handle) = *self.handle.get_mut().unwrap() {
            // SAFETY: the handle was created once and is destroyed exactly
            // once here; nothing touches it afterwards.
            unsafe { ffi::whisperkit_destroy(handle) };
        }
    }
}
