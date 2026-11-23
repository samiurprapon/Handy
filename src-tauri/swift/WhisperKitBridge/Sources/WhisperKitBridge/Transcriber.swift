import Foundation
import WhisperKit

/// Decoding configuration and output shaping.
public enum Transcriber {
    /// Decoding options for a forced-language transcription.
    ///
    /// - task `.transcribe` (never translation)
    /// - `temperature: 0` with zero fallback increments: strictly greedy, the
    ///   Bengali fine-tune was scored this way (see the model card's WER/CER
    ///   methodology) and no randomness is ever introduced.
    /// - prefill prompt on, language detection off: the language is forced
    ///   from the caller, never detected.
    /// - special tokens skipped, no timestamps.
    public static func decodeOptions(language: String?) -> DecodingOptions {
        DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: language,
            temperature: 0,
            temperatureIncrementOnFallback: 0,
            temperatureFallbackCount: 0,
            usePrefillPrompt: true,
            detectLanguage: false,
            skipSpecialTokens: true,
            withoutTimestamps: true
        )
    }

    /// Concatenate the per-window texts WhisperKit returns for long audio
    /// (one `TranscriptionResult` per 30 s window). Each text is trimmed and
    /// the non-empty parts joined with a single space — Bengali script uses
    /// spaces between words, so this matches how a single-window result reads.
    public static func joinTexts(_ results: [TranscriptionResult]) -> String {
        results
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Throw `BridgeError.cancelled` when a cancel was requested.
    public static func checkCancelled(_ flag: Bool) throws {
        if flag {
            throw BridgeError.cancelled
        }
    }
}
