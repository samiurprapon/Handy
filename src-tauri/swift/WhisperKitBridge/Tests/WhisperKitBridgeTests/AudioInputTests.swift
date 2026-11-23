import XCTest
@testable import WhisperKitBridge

final class AudioInputTests: XCTestCase {
    func testEmptyBufferThrows() {
        var empty: [Float] = []
        XCTAssertThrowsError(try empty.withUnsafeBufferPointer {
            try AudioInput.copy(from: $0.baseAddress, count: 0)
        }) { error in
            guard case BridgeError.invalidAudio = error else {
                return XCTFail("expected invalidAudio, got \(error)")
            }
        }
    }

    func testNullPointerThrows() {
        XCTAssertThrowsError(try AudioInput.copy(from: nil, count: 10)) { error in
            guard case BridgeError.invalidAudio = error else {
                return XCTFail("expected invalidAudio, got \(error)")
            }
        }
    }

    func testNonFiniteThrows() {
        let samples: [Float] = [0.1, .nan, 0.3]
        XCTAssertThrowsError(try AudioInput.validate(samples)) { error in
            guard case BridgeError.invalidAudio = error else {
                return XCTFail("expected invalidAudio, got \(error)")
            }
        }
    }

    func testSamplesPassThroughUntouched() throws {
        // No normalization, no clipping, no resampling: exact bits back.
        let samples: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0, 0.000001, 3.4e38]
        let copied = try AudioInput.validate(samples)
        XCTAssertEqual(copied, samples)
    }

    func testTooLongBufferThrows() {
        let samples = [Float](repeating: 0, count: AudioInput.maxSamples + 1)
        XCTAssertThrowsError(try AudioInput.validate(samples)) { error in
            guard case BridgeError.invalidAudio = error else {
                return XCTFail("expected invalidAudio, got \(error)")
            }
        }
    }

    func testVeryShortAudioPasses() throws {
        // A handful of samples (a few milliseconds) is still valid input.
        let copied = try AudioInput.validate([0.25])
        XCTAssertEqual(copied, [0.25])
    }
}
