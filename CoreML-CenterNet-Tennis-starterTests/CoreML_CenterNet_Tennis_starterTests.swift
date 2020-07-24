//
//  CoreML_CenterNet_Tennis_starterTests.swift
//  CoreML-CenterNet-Tennis-starterTests
//
//  Created by Gerald on 7/7/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

import XCTest
import CoreML
import AVFoundation
import Vision

@testable import CoreML_CenterNet_Tennis_starter


class CoreML_CenterNet_Tennis_starterTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    
    // https://stackoverflow.com/questions/28328478/xcode-adding-an-image-to-a-test
    func load_image( name: String, type:String ) throws -> UIImage? {
         let bundle = Bundle(for: CoreML_CenterNet_Tennis_starterTests.self)
         guard let path = bundle.path(forResource: name, ofType: type) else {
             throw NSError(domain: "loadImage", code: 1, userInfo: nil)
         }
         let image = UIImage(contentsOfFile: path)
         return image
    }

    func test_mlmodel() throws {
        
        let bundle = Bundle(for: CoreML_CenterNet_Tennis_starterTests.self)
        let path = bundle.path(forResource: "test_image_outputs.resnet_18_hc64", ofType: "json")

        var expected_outputs: NSArray?
        if let JSONData = try? Data(contentsOf: URL(fileURLWithPath: path!)) {
            expected_outputs = try! JSONSerialization.jsonObject(with: JSONData, options: .mutableContainers) as? NSArray
        }

        let expected_hmap: [[NSNumber]] = expected_outputs![0] as! [[NSNumber]]
        print( "expected hmap: \(expected_hmap.count) \(expected_hmap[0].count)")

        // load the model
        let model = resnet_hc64()
        typealias NetworkInput = resnet_hc64Input
        typealias NetworkOutput = resnet_hc64Output

        let inputName = "input.1"
        let imageConstraint = model.model.modelDescription
            .inputDescriptionsByName[inputName]!
            .imageConstraint!
        let imageOptions: [MLFeatureValue.ImageOption: Any] = [
            .cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
        ]

        
        // FeatureValue via https://github.com/kkk669/coreml-playground/blob/master/CoreML.playgroundbook/Contents/Chapters/CoreML.playgroundchapter/Pages/ObjectRecognitionImage.playgroundpage/main.swift
        let test_image : UIImage? = try? load_image(name: "drop-shot.384", type: "jpg")
        let featureValue: MLFeatureValue = try MLFeatureValue(cgImage: test_image!.cgImage!, constraint: imageConstraint, options: imageOptions)
        
        let inputs: [String: Any] = [
            inputName: featureValue
        ]

        if let provider = try? MLDictionaryFeatureProvider(dictionary: inputs),
            let outFeatures = try? model.model.prediction(from: provider) {
            let result = NetworkOutput(features: outFeatures)

            let output_hmap: MLMultiArray = result.output_hmap
            print( "outputs: \(output_hmap.shape)") // [1, 20, 96, 96]

            // sanity check the shapes of our output
            XCTAssertTrue( Int( truncating: output_hmap.shape[2] ) == expected_hmap.count,
                           "incorrect shape[2]! \(output_hmap.shape[2]) \(expected_hmap.count)" )
            XCTAssertTrue( Int( truncating: output_hmap.shape[3] ) == expected_hmap[0].count,
                           "incorrect shape[3]! \(output_hmap.shape[3]) \(expected_hmap[0].count)" )

            // compare every element of our spectrogram with those from the JSON file
            for i in 0..<expected_hmap.count {
                let spec_row = expected_hmap[i] as [NSNumber]

                for j in 0..<spec_row.count {
                    // only check class 14 (person)
                    let test_idx: [NSNumber] = [ 0, 14, NSNumber(value: i), NSNumber(value: j) ]
                    let val = output_hmap[ test_idx ].floatValue
                    XCTAssertLessThan( abs( val - spec_row[j].floatValue ), 1.0,
                                       "spec vals different at \(i) \(j)! \(val), \(spec_row[j].floatValue)" )
                }
            }
        }
        

    }
    
    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        let model = resnet_hc64()
        let test_image : UIImage? = try? load_image(name: "drop-shot.384", type: "jpg")
        
        let inputName = "input.1"
        let imageConstraint = model.model.modelDescription
            .inputDescriptionsByName[inputName]!
            .imageConstraint!

        let featureValue: MLFeatureValue = try MLFeatureValue(cgImage: test_image!.cgImage!, constraint: imageConstraint)
        
        let inputs: [String: Any] = [
            "input.1": featureValue
        ]


        self.measure {
            // Put the code you want to measure the time of here.
            
            let provider = try? MLDictionaryFeatureProvider(dictionary: inputs)
            let _ = try? model.model.prediction(from: provider!)
            // resnet18_hc64: average: 0.408
        }
    }

}
