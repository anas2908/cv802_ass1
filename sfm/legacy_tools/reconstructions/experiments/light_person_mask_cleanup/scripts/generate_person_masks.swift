import Foundation
import Vision
import ImageIO
import CoreGraphics
import CoreVideo
import UniformTypeIdentifiers

let manifestURL=URL(fileURLWithPath:CommandLine.arguments[1])
let rows=try JSONSerialization.jsonObject(with:Data(contentsOf:manifestURL)) as! [[String:String]]
let manager=FileManager.default
for (index,row) in rows.enumerated() {
    autoreleasepool {
        let rawURL=URL(fileURLWithPath:row["raw_mask"]!)
        let metadataURL=URL(fileURLWithPath:row["raw_metadata"]!)
        do {
            try manager.createDirectory(at:rawURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            try manager.createDirectory(at:metadataURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:row["source"]!) as CFURL,nil),
                  let image=CGImageSourceCreateImageAtIndex(source,0,nil) else {
                throw NSError(domain:"MaskPreparation",code:1,userInfo:[NSLocalizedDescriptionKey:"Source image decode failed"])
            }
            let properties=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [String:Any]
            let orientation=properties?[kCGImagePropertyOrientation as String] as? Int ?? 1
            guard orientation==1 else {
                throw NSError(domain:"MaskPreparation",code:2,userInfo:[NSLocalizedDescriptionKey:"Input is not physically upright orientation1"])
            }
            // Independent request per still/frame avoids temporal state between unrelated viewpoints.
            let request=VNGeneratePersonSegmentationRequest()
            request.qualityLevel = .accurate
            request.outputPixelFormat = kCVPixelFormatType_OneComponent8
            request.preferBackgroundProcessing = true
            let handler=VNImageRequestHandler(cgImage:image,orientation:.up,options:[:])
            try handler.perform([request])
            guard let observation=request.results?.first else {
                throw NSError(domain:"MaskPreparation",code:3,userInfo:[NSLocalizedDescriptionKey:"Vision returned no segmentation observation"])
            }
            let buffer=observation.pixelBuffer
            CVPixelBufferLockBaseAddress(buffer,.readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer,.readOnly) }
            let width=CVPixelBufferGetWidth(buffer),height=CVPixelBufferGetHeight(buffer),stride=CVPixelBufferGetBytesPerRow(buffer)
            guard CVPixelBufferGetPixelFormatType(buffer)==kCVPixelFormatType_OneComponent8,
                  let base=CVPixelBufferGetBaseAddress(buffer) else {
                throw NSError(domain:"MaskPreparation",code:4,userInfo:[NSLocalizedDescriptionKey:"Unexpected mask pixel format or missing buffer"])
            }
            let data=Data(bytes:base,count:stride*height)
            guard let provider=CGDataProvider(data:data as CFData),
                  let mask=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:8,bytesPerRow:stride,space:CGColorSpaceCreateDeviceGray(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent),
                  let destination=CGImageDestinationCreateWithURL(rawURL as CFURL,UTType.png.identifier as CFString,1,nil) else {
                throw NSError(domain:"MaskPreparation",code:5,userInfo:[NSLocalizedDescriptionKey:"Could not create grayscale PNG"])
            }
            CGImageDestinationAddImage(destination,mask,[kCGImagePropertyOrientation:1] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else {
                throw NSError(domain:"MaskPreparation",code:6,userInfo:[NSLocalizedDescriptionKey:"Writing mask PNG failed"])
            }
            var metadata:[String:Any]=row
            metadata["status"]="generated"
            metadata["source_width"]=image.width;metadata["source_height"]=image.height
            metadata["source_orientation"]=orientation;metadata["handler_orientation"]="up"
            metadata["raw_mask_width"]=width;metadata["raw_mask_height"]=height
            metadata["raw_pixel_format"]="OneComponent8";metadata["quality_level"]="accurate"
            metadata["request_revision"]=request.revision;metadata["independent_request_per_image"]=true
            metadata["macos_version"]=ProcessInfo.processInfo.operatingSystemVersionString
            try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys]).write(to:metadataURL,options:.atomic)
        } catch {
            var metadata:[String:Any]=row
            metadata["status"]="error";metadata["error"]=String(describing:error)
            if let output=try? JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys]) {
                try? output.write(to:metadataURL,options:.atomic)
            }
            print("FAILED \(row["image_name"]!): \(error)")
        }
    }
    print("Segmentation \(index+1)/\(rows.count): \(row["image_name"]!)");fflush(stdout)
}
