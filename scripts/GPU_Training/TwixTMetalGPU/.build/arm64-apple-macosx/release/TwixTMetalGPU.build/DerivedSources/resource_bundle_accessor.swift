import Foundation

extension Foundation.Bundle {
    static let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("TwixTMetalGPU_TwixTMetalGPU.bundle").path
        let buildPath = "/Users/bill/Desktop/TwixT_Game/scripts/GPU_Training/TwixTMetalGPU/.build/arm64-apple-macosx/release/TwixTMetalGPU_TwixTMetalGPU.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}