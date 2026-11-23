import Foundation
import WhisperKit

/// Loads and validates WhisperKit model folders.
///
/// Handy downloads the model files itself (progress, cancellation, and
/// resumption live in the Rust layer); this only checks that what landed on
/// disk is a complete WhisperKit model folder before handing it to WhisperKit.
public enum ModelLoader {
    /// Files every WhisperKit conversion must contain. The `.mlmodelc`
    /// bundles are directories; `model.mil` is their load-bearing content.
    public static let requiredPaths: [String] = [
        "config.json",
        "tokenizer.json",
        "AudioEncoder.mlmodelc/model.mil",
        "MelSpectrogram.mlmodelc/model.mil",
        "TextDecoder.mlmodelc/model.mil",
    ]

    /// Throws `BridgeError.invalidModelFolder` listing everything missing.
    public static func validateFolder(_ url: URL) throws {
        let missing = requiredPaths.filter { path in
            !FileManager.default.fileExists(atPath: url.appendingPathComponent(path).path)
        }
        if !missing.isEmpty {
            throw BridgeError.invalidModelFolder(missing)
        }
    }

    /// Compute placement verified against the Bengali Whisper Medium Core ML
    /// conversion (omnixima/bengaliai-whisper-medium-coreml): the encoder and
    /// decoder run their palettized weights best on the Neural Engine with
    /// CPU fallback (`.cpuAndNeuralEngine` lets Core ML fall back per-op; it
    /// never forces an unsupported unit), and the Mel spectrogram and prefill
    /// path are cheap enough that CPU-only avoids needless GPU/ANE dispatch
    /// overhead for tens-of-milliseconds of work.
    public static func computeOptions() -> ModelComputeOptions {
        ModelComputeOptions(
            melCompute: .cpuOnly,
            audioEncoderCompute: .cpuAndNeuralEngine,
            textDecoderCompute: .cpuAndNeuralEngine
        )
    }

    /// A `WhisperKitConfig` for an already-downloaded model folder. Downloads
    /// are always off (`false`): Rust owns fetching.
    public static func config(folder: URL) -> WhisperKitConfig {
        WhisperKitConfig(
            modelFolder: folder.path,
            computeOptions: computeOptions(),
            verbose: false,
            logLevel: .error,
            prewarm: false,
            load: true,
            download: false
        )
    }
}
