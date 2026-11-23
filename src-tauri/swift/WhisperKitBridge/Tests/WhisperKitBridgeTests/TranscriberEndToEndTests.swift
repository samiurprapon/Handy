import XCTest
@testable import WhisperKitBridge

/// End-to-end tests against the real model and the FFI boundary.
///
/// Skipped unless WHISPERKIT_E2E points at a downloaded model folder, e.g.
/// after Handy downloads it:
///
///     WHISPERKIT_E2E="$HOME/Library/Application Support/com.pais.handy/models/whisperkit/bengaliai-whisper-medium" \
///         swift test --package-path src-tauri/swift/WhisperKitBridge
final class TranscriberEndToEndTests: XCTestCase {
    private var modelFolder: URL? {
        ProcessInfo.processInfo.environment["WHISPERKIT_E2E"].map {
            URL(fileURLWithPath: $0)
        }
    }

    func testLoadTranscribeBengaliUnload() async throws {
        guard let folder = modelFolder else {
            throw XCTSkip("WHISPERKIT_E2E not set; skipping live model test")
        }
        try ModelLoader.validateFolder(folder)
        let manager = WhisperKitManager()
        try await manager.load(folder: folder)
        do {
            let loaded = await manager.isLoaded
            XCTAssertTrue(loaded)

            // 1 s of silence: decodes to empty (or near-empty) text without
            // erroring — the no-speech path works.
            let silence = [Float](repeating: 0, count: 16_000)
            let quiet = try await manager.transcribe(samples: silence, language: "bn")
            XCTAssertTrue(quiet.trimmingCharacters(in: .whitespaces).count < 8)

            // Very short audio must not crash the decode.
            let blip = try await manager.transcribe(samples: [0.1, 0.2, 0.1], language: "bn")
            _ = blip
        }
        await manager.unload()
        let stillLoaded = await manager.isLoaded
        XCTAssertFalse(stillLoaded)
    }

    func testTranscribeWithoutModelThrowsNotLoaded() async {
        let manager = WhisperKitManager()
        do {
            _ = try await manager.transcribe(samples: [0.5], language: "bn")
            XCTFail("expected notLoaded")
        } catch let error as BridgeError {
            guard case BridgeError.notLoaded = error else {
                return XCTFail("expected notLoaded, got \(error)")
            }
        } catch {
            XCTFail("unexpected error type \(error)")
        }
    }
}
