//! Rust side of the WhisperKit (Swift/Core ML) transcription engine.
//!
//! Real engine on `macos + aarch64` when the Swift package built; a stub with
//! the identical API everywhere else (and on builds where the toolchain could
//! not compile WhisperKit), so platform code paths compile unchanged and
//! report "unsupported" at runtime.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod engine;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod ffi;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod stub;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use engine::WhisperKitEngine;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub use stub::WhisperKitEngine;

pub use error::WhisperKitError;

mod error {
    /// Errors from the WhisperKit engine, mirroring the Swift `BridgeError`
    /// cases plus the build-level availability state.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum WhisperKitError {
        /// This build cannot run WhisperKit (wrong platform, or the Swift
        /// package could not be compiled).
        NotSupported,
        /// No model is loaded.
        NotLoaded,
        /// Loading failed (invalid folder, Core ML compile failure, ...).
        LoadFailed(String),
        /// The audio buffer was rejected.
        InvalidAudio(String),
        /// The decode failed.
        TranscribeFailed(String),
        /// The run was cancelled via [`WhisperKitEngine::cancel`].
        Cancelled,
    }

    impl std::fmt::Display for WhisperKitError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::NotSupported => write!(
                    f,
                    "WhisperKit is not available in this build (requires macOS \
                     on Apple Silicon built with the WhisperKit Swift package)"
                ),
                Self::NotLoaded => write!(f, "No WhisperKit model is loaded"),
                Self::LoadFailed(m) => write!(f, "{m}"),
                Self::InvalidAudio(m) => write!(f, "Invalid audio: {m}"),
                Self::TranscribeFailed(m) => write!(f, "{m}"),
                Self::Cancelled => write!(f, "cancelled"),
            }
        }
    }

    impl std::error::Error for WhisperKitError {}
}
