import Foundation

/// The C FFI surface of the bridge.
///
/// Contract (mirrored on the Rust side in `src-tauri/src/whisperkit/ffi.rs`):
///
/// - An opaque handle is returned by `whisperkit_create` (retained) and must
///   be released exactly once with `whisperkit_destroy`. Every other function
///   borrows it without touching its refcount.
/// - The caller serializes all calls on one handle (Rust holds a mutex); the
///   actor inside additionally serializes the async work.
/// - These functions BLOCK until the requested work completes. Callers must
///   never invoke them from the process main thread or a Swift concurrency
///   cooperative-pool thread — Rust calls them from its own worker threads.
/// - Every fallible function returns a heap-allocated `WhisperKitResult *`
///   (never NULL) with exactly one of `text`/`error` set; the caller frees it
///   with `whisperkit_free_result`. All strings are heap-allocated UTF-8
///   owned by the caller after return.
/// - No Swift traps cross the boundary: every input is validated and every
///   error is converted to a result string.

/// The WhisperKit release this bridge is pinned to (Package.swift).
public let whisperKitVersion = "1.1.1"

/// C-layout result struct. Rust declares the mirror `#[repr(C)]` struct.
public struct WhisperKitResult {
    public var text: UnsafeMutablePointer<CChar>?
    public var error: UnsafeMutablePointer<CChar>?
}

/// Allocate a success result (`text` set, `error` NULL).
private func resultSuccess(_ text: String) -> UnsafeMutableRawPointer {
    let pointer = UnsafeMutablePointer<WhisperKitResult>.allocate(capacity: 1)
    pointer.initialize(to: WhisperKitResult(text: strdup(text), error: nil))
    return UnsafeMutableRawPointer(pointer)
}

/// Allocate a failure result (`error` set, `text` NULL).
private func resultFailure(_ message: String) -> UnsafeMutableRawPointer {
    let pointer = UnsafeMutablePointer<WhisperKitResult>.allocate(capacity: 1)
    pointer.initialize(to: WhisperKitResult(text: nil, error: strdup(message)))
    return UnsafeMutableRawPointer(pointer)
}

private func resultMap(_ body: () throws -> String) -> UnsafeMutableRawPointer {
    do {
        return resultSuccess(try body())
    } catch {
        return resultFailure(BridgeErrors.message(error))
    }
}

/// The object behind the opaque handle.
final class WhisperKitHandle {
    let manager = WhisperKitManager()
}

/// `Result`'s failure side must conform to `Error`; strings do not.
struct FailureBox: Error {
    let message: String
}

/// Lock-guarded single-assignment cell, safe across the semaphore handshake.
private final class ResultStorage<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, FailureBox>?

    func store(_ result: Result<T, FailureBox>) {
        lock.lock()
        self.result = result
        lock.unlock()
    }

    func load() -> Result<T, FailureBox>? {
        lock.lock()
        defer { lock.unlock() }
        return result
    }
}

// MARK: - C entry points

@_cdecl("whisperkit_supported")
public func whisperkitSupported() -> Int32 {
    // The shipped Bengali conversion carries palettized weights targeting
    // the iOS 17 / macOS 14 Core ML runtime; older systems cannot load them.
    if #available(macOS 14.0, *) {
        return 1
    }
    return 0
}

/// Allocated once (global lets initialize lazily and thread-safely) and never
/// freed, so the pointer stays valid for the process lifetime.
private let versionCString: UnsafeMutablePointer<CChar> =
    strdup("WhisperKit \(whisperKitVersion)")!

@_cdecl("whisperkit_bridge_version")
public func whisperkitBridgeVersion() -> UnsafePointer<CChar> {
    UnsafePointer(versionCString)
}

@_cdecl("whisperkit_create")
public func whisperkitCreate() -> UnsafeMutableRawPointer? {
    Unmanaged.passRetained(WhisperKitHandle()).toOpaque()
}

@_cdecl("whisperkit_destroy")
public func whisperkitDestroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeRetainedValue()
    // Release model memory before the last reference goes away.
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        await box.manager.unload()
        semaphore.signal()
    }
    semaphore.wait()
}

@_cdecl("whisperkit_load_model")
public func whisperkitLoadModel(
    _ handle: UnsafeMutableRawPointer,
    _ modelPath: UnsafePointer<CChar>
) -> UnsafeMutableRawPointer {
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeUnretainedValue()
    let folder = URL(fileURLWithPath: String(cString: modelPath))
    let semaphore = DispatchSemaphore(value: 0)
    let storage = ResultStorage<String>()
    Task.detached {
        do {
            try await box.manager.load(folder: folder)
            storage.store(.success(""))
        } catch {
            storage.store(.failure(FailureBox(message: BridgeErrors.message(error))))
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = storage.load() else {
        return resultFailure("unreachable: semaphore released before value stored")
    }
    switch result {
    case .success:
        return resultSuccess("")
    case let .failure(box):
        return resultFailure(box.message)
    }
}

@_cdecl("whisperkit_unload_model")
public func whisperkitUnloadModel(
    _ handle: UnsafeMutableRawPointer
) -> UnsafeMutableRawPointer {
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeUnretainedValue()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        await box.manager.unload()
        semaphore.signal()
    }
    semaphore.wait()
    return resultSuccess("")
}

@_cdecl("whisperkit_is_model_loaded")
public func whisperkitIsModelLoaded(_ handle: UnsafeMutableRawPointer) -> Int32 {
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeUnretainedValue()
    let semaphore = DispatchSemaphore(value: 0)
    var loaded = false
    Task.detached {
        loaded = await box.manager.isLoaded
        semaphore.signal()
    }
    semaphore.wait()
    return loaded ? 1 : 0
}

@_cdecl("whisperkit_transcribe")
public func whisperkitTranscribe(
    _ handle: UnsafeMutableRawPointer,
    _ samples: UnsafePointer<Float>?,
    _ count: Int,
    _ language: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer {
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeUnretainedValue()
    let language = language.map { String(cString: $0) }
    let ownedSamples: [Float]
    do {
        ownedSamples = try AudioInput.copy(from: samples, count: count)
    } catch {
        return resultFailure(BridgeErrors.message(error))
    }
    let semaphore = DispatchSemaphore(value: 0)
    let storage = ResultStorage<String>()
    Task.detached {
        do {
            let text = try await box.manager.transcribe(
                samples: ownedSamples, language: language)
            storage.store(.success(text))
        } catch {
            storage.store(.failure(FailureBox(message: BridgeErrors.message(error))))
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = storage.load() else {
        return resultFailure("unreachable: semaphore released before value stored")
    }
    switch result {
    case let .success(text):
        return resultSuccess(text)
    case let .failure(box):
        return resultFailure(box.message)
    }
}

@_cdecl("whisperkit_cancel")
public func whisperkitCancel(_ handle: UnsafeMutableRawPointer) {
    let box = Unmanaged<WhisperKitHandle>.fromOpaque(handle).takeUnretainedValue()
    // Fire-and-forget: cancel must never block the caller behind a
    // transcription in flight.
    Task { await box.manager.cancel() }
}

@_cdecl("whisperkit_free_result")
public func whisperkitFreeResult(_ result: UnsafeMutableRawPointer?) {
    guard let result else { return }
    let pointer = result.assumingMemoryBound(to: WhisperKitResult.self)
    free(pointer.pointee.text)
    free(pointer.pointee.error)
    pointer.deinitialize(count: 1)
    pointer.deallocate()
}
