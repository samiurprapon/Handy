import Foundation

// Stub implementation when the WhisperKit Swift package cannot be built
// (no full toolchain, forced via HANDY_FORCE_WHISPERKIT_STUB, or the package
// build failed). Same symbols as WhisperKitBridge/Sources/WhisperKitBridge/
// Bridge.swift; every call fails and whisperkit_supported() reports 0 so the
// Rust side hides the engine and its models.

struct WhisperKitResult {
    var text: UnsafeMutablePointer<CChar>?
    var error: UnsafeMutablePointer<CChar>?
}

private func failure(_ message: String) -> UnsafeMutableRawPointer {
    let pointer = UnsafeMutablePointer<WhisperKitResult>.allocate(capacity: 1)
    pointer.initialize(to: WhisperKitResult(text: nil, error: strdup(message)))
    return UnsafeMutableRawPointer(pointer)
}

@_cdecl("whisperkit_supported")
public func whisperkitSupported() -> Int32 { 0 }

@_cdecl("whisperkit_bridge_version")
public func whisperkitBridgeVersion() -> UnsafePointer<CChar> {
    // Lives for the process lifetime; never freed.
    UnsafePointer(versionCString)
}

private let versionCString: UnsafeMutablePointer<CChar> =
    strdup("WhisperKit unavailable (stub build)")!

@_cdecl("whisperkit_create")
public func whisperkitCreate() -> UnsafeMutableRawPointer? { nil }

@_cdecl("whisperkit_destroy")
public func whisperkitDestroy(_ handle: UnsafeMutableRawPointer?) {}

@_cdecl("whisperkit_load_model")
public func whisperkitLoadModel(
    _ handle: UnsafeMutableRawPointer,
    _ modelPath: UnsafePointer<CChar>
) -> UnsafeMutableRawPointer {
    failure("WhisperKit is not available in this build (Swift package not compiled).")
}

@_cdecl("whisperkit_unload_model")
public func whisperkitUnloadModel(
    _ handle: UnsafeMutableRawPointer
) -> UnsafeMutableRawPointer {
    failure("WhisperKit is not available in this build (Swift package not compiled).")
}

@_cdecl("whisperkit_is_model_loaded")
public func whisperkitIsModelLoaded(_ handle: UnsafeMutableRawPointer) -> Int32 { 0 }

@_cdecl("whisperkit_transcribe")
public func whisperkitTranscribe(
    _ handle: UnsafeMutableRawPointer,
    _ samples: UnsafePointer<Float>?,
    _ count: Int,
    _ language: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer {
    failure("WhisperKit is not available in this build (Swift package not compiled).")
}

@_cdecl("whisperkit_cancel")
public func whisperkitCancel(_ handle: UnsafeMutableRawPointer) {}

@_cdecl("whisperkit_free_result")
public func whisperkitFreeResult(_ result: UnsafeMutableRawPointer?) {
    guard let result else { return }
    let pointer = result.assumingMemoryBound(to: WhisperKitResult.self)
    free(pointer.pointee.text)
    free(pointer.pointee.error)
    pointer.deinitialize(count: 1)
    pointer.deallocate()
}
