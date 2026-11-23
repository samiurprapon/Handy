import XCTest
import WhisperKit
@testable import WhisperKitBridge

final class TranscriberTests: XCTestCase {
    func testDecodeOptionsForceBengali() {
        let options = Transcriber.decodeOptions(language: "bn")
        XCTAssertEqual(options.task, .transcribe)
        XCTAssertEqual(options.language, "bn")
        XCTAssertEqual(options.temperature, 0)
        XCTAssertEqual(options.temperatureIncrementOnFallback, 0)
        XCTAssertEqual(options.temperatureFallbackCount, 0)
        XCTAssertTrue(options.usePrefillPrompt)
        XCTAssertFalse(options.detectLanguage)
        XCTAssertTrue(options.skipSpecialTokens)
        XCTAssertTrue(options.withoutTimestamps)
        XCTAssertFalse(options.wordTimestamps)
        XCTAssertNil(options.chunkingStrategy)
    }

    func testDecodeOptionsWithoutLanguage() {
        // A nil language still disables detection: the caller decides.
        let options = Transcriber.decodeOptions(language: nil)
        XCTAssertNil(options.language)
        XCTAssertFalse(options.detectLanguage)
    }

    func testComputeOptionsPreferNeuralEngine() {
        let options = ModelLoader.computeOptions()
        XCTAssertEqual(options.melCompute, .cpuOnly)
        XCTAssertEqual(options.audioEncoderCompute, .cpuAndNeuralEngine)
        XCTAssertEqual(options.textDecoderCompute, .cpuAndNeuralEngine)
    }

    func testCancelledFlagThrows() {
        XCTAssertThrowsError(try Transcriber.checkCancelled(true)) { error in
            guard case BridgeError.cancelled = error else {
                return XCTFail("expected cancelled, got \(error)")
            }
        }
        XCTAssertNoThrow(try Transcriber.checkCancelled(false))
    }

    func testErrorMessageMapping() {
        XCTAssertEqual(BridgeErrors.message(BridgeError.cancelled), "cancelled")
        XCTAssertEqual(BridgeErrors.message(CancellationError()), "cancelled")
        XCTAssertTrue(
            BridgeErrors.message(BridgeError.notLoaded)
                .contains("No WhisperKit model"))
        // Unknown Swift errors produce a non-empty string.
        struct Weird: Error {}
        XCTAssertFalse(BridgeErrors.message(Weird()).isEmpty)
    }

    func testBridgeVersionMatchesPin() {
        // Keep in lockstep with Package.swift's exact pin.
        XCTAssertEqual(whisperKitVersion, "1.1.1")
    }
}
