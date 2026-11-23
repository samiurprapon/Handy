import Foundation
import WhisperKit

/// Owns the single WhisperKit instance. An actor: the FFI boundary is called
/// from arbitrary threads (Rust worker threads, never a Swift concurrency
/// thread), and this serializes load / transcribe / unload so two recordings
/// can never share the model mid-flight.
///
/// The instance is retained across recordings — model weights load once and
/// are released only on explicit unload (or destroy, which unloads first).
public actor WhisperKitManager {
    private var whisperKit: WhisperKit?
    private var currentTranscription: Task<[TranscriptionResult], any Error>?
    private var cancelRequested = false

    public init() {}

    public var isLoaded: Bool {
        whisperKit != nil
    }

    /// Load `folder`, replacing any loaded model. The first load of a model
    /// on a device may take ~a minute while Core ML compiles it for the
    /// Neural Engine; later loads are fast (the compiled artifact is cached
    /// by the OS).
    public func load(folder: URL) async throws {
        cancelRequested = false
        try await unloadCurrent()
        try ModelLoader.validateFolder(folder)
        let kit = try await WhisperKit(ModelLoader.config(folder: folder))
        whisperKit = kit
    }

    /// Release the model and its compiled state.
    public func unload() async {
        try? await unloadCurrent()
    }

    private func unloadCurrent() async throws {
        if let kit = whisperKit {
            await kit.unloadModels()
        }
        whisperKit = nil
    }

    /// Request cancellation of the in-flight transcription. Returns
    /// immediately; the running decode observes it at its next await point
    /// (or on completion, where the result is discarded).
    public func cancel() {
        // Only an in-flight run can be cancelled. Handy cancels on every
        // dictation abort (often while still recording); a flag set with
        // nothing running would poison the next transcription.
        guard let task = currentTranscription else { return }
        cancelRequested = true
        task.cancel()
    }

    /// Transcribe `samples` (16 kHz mono Float32) with `language` forced.
    /// Returns the joined text of all decode windows.
    public func transcribe(samples: [Float], language: String?) async throws -> String {
        guard let kit = whisperKit else {
            throw BridgeError.notLoaded
        }
        cancelRequested = false
        let options = Transcriber.decodeOptions(language: language)
        let task: Task<[TranscriptionResult], any Error> = Task {
            try await kit.transcribe(audioArray: samples, decodeOptions: options)
        }
        currentTranscription = task
        defer {
            currentTranscription = nil
            cancelRequested = false
        }
        do {
            let results = try await task.value
            // A cancel that arrived while decoding: discard the result.
            try Transcriber.checkCancelled(cancelRequested)
            return Transcriber.joinTexts(results)
        } catch is CancellationError {
            throw BridgeError.cancelled
        }
    }
}
