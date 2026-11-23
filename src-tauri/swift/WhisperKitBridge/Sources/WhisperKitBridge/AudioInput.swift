import Foundation

/// Validation of the PCM handed over the FFI boundary.
///
/// Handy records and preprocesses audio itself (16 kHz mono Float32, VAD
/// applied upstream), so this only rejects buffers WhisperKit cannot consume.
/// Samples are never normalized, clipped, or resampled here — they pass
/// through untouched.
public enum AudioInput {
    public static let sampleRate: Int = 16_000

    /// Upper bound accepted from the boundary (2 hours). Longer input is a
    /// caller bug and would otherwise pin memory inside WhisperKit.
    public static let maxSamples: Int = sampleRate * 60 * 120

    /// Copy `count` samples from `pointer` into an owned array, rejecting
    /// buffers WhisperKit cannot decode.
    public static func copy(from pointer: UnsafePointer<Float>?, count: Int) throws -> [Float] {
        guard let pointer else {
            throw BridgeError.invalidAudio("null sample pointer")
        }
        guard count > 0 else {
            throw BridgeError.invalidAudio("empty buffer (0 samples)")
        }
        guard count <= maxSamples else {
            throw BridgeError.invalidAudio(
                "buffer too long (\(count) samples > \(maxSamples))")
        }
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        let samples = [Float](buffer)
        if let index = samples.firstIndex(where: { !$0.isFinite }) {
            throw BridgeError.invalidAudio("non-finite sample at index \(index)")
        }
        return samples
    }

    /// Same validation for an already-owned array (used by tests and the
    /// Swift-level API).
    public static func validate(_ samples: [Float]) throws -> [Float] {
        try samples.withUnsafeBufferPointer { buffer in
            try copy(from: buffer.baseAddress, count: samples.count)
        }
    }
}
