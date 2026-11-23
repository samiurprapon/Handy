//! Raw FFI declarations for the WhisperKitBridge Swift package.
//!
//! # Safety invariants
//!
//! - `whisperkit_create` returns a retained handle; it must be passed to
//!   `whisperkit_destroy` exactly once, and to nothing else afterwards.
//! - Calls on one handle may overlap (`whisperkit_cancel` must reach an
//!   in-flight `whisperkit_transcribe`); the Swift actor inside serializes
//!   the work.
//! - Fallible calls return a heap pointer that must be freed with
//!   `whisperkit_free_result`, exactly once, and never read afterwards.
//! - These functions block for as long as the Swift work takes; never call
//!   them from the Tauri main thread or an async runtime worker — callers use
//!   dedicated threads.
//! - Swift never traps across the boundary: every failure arrives as an
//!   error string inside the result struct.

use std::ffi::c_char;

/// Mirror of the Swift `WhisperKitResult` struct (two optional heap strings).
#[repr(C)]
pub struct WhisperKitResult {
    pub text: *mut c_char,
    pub error: *mut c_char,
}

extern "C" {
    /// 1 when the runtime (OS version + build) can run WhisperKit.
    pub fn whisperkit_supported() -> i32;
    /// Static version string of the pinned WhisperKit release; never freed.
    pub fn whisperkit_bridge_version() -> *const c_char;
    /// Create the engine handle (retained). Never fails.
    pub fn whisperkit_create() -> *mut std::ffi::c_void;
    /// Load the model folder at `model_path` (UTF-8). Blocking.
    pub fn whisperkit_load_model(
        handle: *mut std::ffi::c_void,
        model_path: *const c_char,
    ) -> *mut WhisperKitResult;
    /// Unload the current model, releasing its memory. Blocking.
    pub fn whisperkit_unload_model(handle: *mut std::ffi::c_void) -> *mut WhisperKitResult;
    /// 1 while a model is loaded. Blocking (actor hop).
    pub fn whisperkit_is_model_loaded(handle: *mut std::ffi::c_void) -> i32;
    /// Transcribe `count` 16 kHz mono f32 samples. Blocking. `language` may
    /// be NULL.
    pub fn whisperkit_transcribe(
        handle: *mut std::ffi::c_void,
        samples: *const f32,
        count: usize,
        language: *const c_char,
    ) -> *mut WhisperKitResult;
    /// Request cancellation of the in-flight transcription. Non-blocking.
    pub fn whisperkit_cancel(handle: *mut std::ffi::c_void);
    /// Free a result and both of its strings. NULL is a no-op.
    pub fn whisperkit_free_result(result: *mut WhisperKitResult);
    /// Release the handle exactly once. Unloads any model first. Blocking.
    pub fn whisperkit_destroy(handle: *mut std::ffi::c_void);
}
