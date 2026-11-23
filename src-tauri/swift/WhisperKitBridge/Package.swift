// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "WhisperKitBridge",
    platforms: [
        // WhisperKit's own floor. The app's minimum (10.15) is lower; the
        // bridge's FFI entry points runtime-gate on macOS 14 (the palettized
        // Core ML weights used by the Bengali model need it), so nothing under
        // 13 ever reaches this code.
        .macOS(.v13)
    ],
    dependencies: [
        // Pinned exact: DecodingOptions / ModelComputeOptions / prefill
        // behavior verified against this release. Bump deliberately and
        // re-verify (see Tests/WhisperKitBridgeTests/TranscriberTests.swift).
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", exact: "1.1.1"),
    ],
    targets: [
        .target(
            name: "WhisperKitBridge",
            dependencies: [
                .product(name: "WhisperKit", package: "WhisperKit"),
            ]
        ),
        .testTarget(
            name: "WhisperKitBridgeTests",
            dependencies: ["WhisperKitBridge"]
        ),
    ]
)
