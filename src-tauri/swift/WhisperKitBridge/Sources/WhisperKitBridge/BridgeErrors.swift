import Foundation

/// Errors surfaced through the FFI boundary. Every case maps to a stable,
/// human-readable message; Rust turns the string into its own error type.
public enum BridgeError: Error {
    /// The host or build cannot run WhisperKit (OS too old, stub build).
    case notSupported(String)
    /// No model is loaded.
    case notLoaded
    /// The model folder is missing required files.
    case invalidModelFolder([String])
    /// The PCM buffer is unusable.
    case invalidAudio(String)
    /// WhisperKit failed mid-decode.
    case transcriptionFailed(String)
    /// The run was cancelled through `whisperkit_cancel`.
    case cancelled
}

extension BridgeError {
    /// Stable message for the C boundary. Keep these strings short and
    /// diagnostic; Handy's notice system shows them verbatim.
    public var message: String {
        switch self {
        case let .notSupported(reason):
            return "WhisperKit is not available: \(reason)"
        case .notLoaded:
            return "No WhisperKit model is loaded."
        case let .invalidModelFolder(missing):
            return "Model folder is incomplete; missing: \(missing.joined(separator: ", "))"
        case let .invalidAudio(reason):
            return "Invalid audio: \(reason)"
        case let .transcriptionFailed(reason):
            return "WhisperKit transcription failed: \(reason)"
        case .cancelled:
            return "cancelled"
        }
    }
}

public enum BridgeErrors {
    /// Best-effort message for any thrown error, Swift or BridgeError.
    public static func message(_ error: Error) -> String {
        if let bridge = error as? BridgeError {
            return bridge.message
        }
        if error is CancellationError {
            return BridgeError.cancelled.message
        }
        if let localized = error as? LocalizedError, let text = localized.errorDescription {
            return text
        }
        return String(describing: error)
    }
}
