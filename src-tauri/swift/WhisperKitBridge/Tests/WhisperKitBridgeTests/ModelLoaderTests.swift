import XCTest
@testable import WhisperKitBridge

final class ModelLoaderTests: XCTestCase {
    private func makeFolder(files: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wkbridge-tests-\(UUID().uuidString)")
        for file in files {
            let url = root.appendingPathComponent(file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data("stub".utf8).write(to: url)
        }
        return root
    }

    override func tearDown() {
        let tmp = FileManager.default.temporaryDirectory
        for entry in (try? FileManager.default.contentsOfDirectory(atPath: tmp.path)) ?? []
        where entry.hasPrefix("wkbridge-tests-") {
            try? FileManager.default.removeItem(at: tmp.appendingPathComponent(entry))
        }
    }

    func testCompleteFolderValidates() throws {
        let folder = try makeFolder(files: ModelLoader.requiredPaths)
        XCTAssertNoThrow(try ModelLoader.validateFolder(folder))
    }

    func testEmptyFolderListsEveryMissingFile() {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("wkbridge-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        XCTAssertThrowsError(try ModelLoader.validateFolder(folder)) { error in
            guard case let BridgeError.invalidModelFolder(missing) = error else {
                return XCTFail("expected invalidModelFolder, got \(error)")
            }
            XCTAssertEqual(Set(missing), Set(ModelLoader.requiredPaths))
        }
    }

    func testPartialFolderNamesTheMissingPieces() throws {
        let present = ["config.json", "tokenizer.json"]
        let folder = try makeFolder(files: present)
        XCTAssertThrowsError(try ModelLoader.validateFolder(folder)) { error in
            guard case let BridgeError.invalidModelFolder(missing) = error else {
                return XCTFail("expected invalidModelFolder, got \(error)")
            }
            XCTAssertEqual(
                Set(missing),
                Set(ModelLoader.requiredPaths.filter { !present.contains($0) }))
        }
    }

    func testConfigNeverDownloads() {
        let config = ModelLoader.config(
            folder: URL(fileURLWithPath: "/nonexistent/bengaliai-whisper-medium"))
        XCTAssertFalse(config.download)
        XCTAssertEqual(config.modelFolder, "/nonexistent/bengaliai-whisper-medium")
        XCTAssertNil(config.modelRepo)
    }
}
