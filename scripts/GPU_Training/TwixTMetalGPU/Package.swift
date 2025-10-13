// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TwixTMetalGPU",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(
            name: "twixt-metal-worker",
            targets: ["TwixTMetalGPU"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "TwixTMetalGPU",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            resources: [
                .process("Shaders")
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("AccessLevelOnImport"),
                .unsafeFlags(["-O", "-whole-module-optimization"], .when(configuration: .release))
            ]
        )
    ]
)
