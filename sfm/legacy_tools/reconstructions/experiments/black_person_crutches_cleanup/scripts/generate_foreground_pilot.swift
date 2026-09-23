import Foundation
import Vision
import ImageIO
import CoreGraphics
import CoreVideo
import UniformTypeIdentifiers

func writeGray(_ buffer:CVPixelBuffer,to url:URL) throws -> [String:Int] {
    CVPixelBufferLockBaseAddress(buffer,.readOnly)
    defer {CVPixelBufferUnlockBaseAddress(buffer,.readOnly)}
    let width=CVPixelBufferGetWidth(buffer),height=CVPixelBufferGetHeight(buffer),stride=CVPixelBufferGetBytesPerRow(buffer)
    guard let base=CVPixelBufferGetBaseAddress(buffer) else {throw NSError(domain:"Mask",code:1)}
    let format=CVPixelBufferGetPixelFormatType(buffer)
    var data=Data(count:width*height)
    data.withUnsafeMutableBytes { destination in
        let bytes=destination.bindMemory(to:UInt8.self)
        for y in 0..<height {
            if format==kCVPixelFormatType_OneComponent8 {
                let source=base.advanced(by:y*stride).assumingMemoryBound(to:UInt8.self)
                for x in 0..<width {bytes[y*width+x]=source[x]}
            } else if format==kCVPixelFormatType_OneComponent32Float {
                let source=base.advanced(by:y*stride).assumingMemoryBound(to:Float.self)
                for x in 0..<width {let value=source[x];bytes[y*width+x]=value.isFinite ? UInt8((max(0,min(1,value))*255).rounded()) : 0}
            }
        }
    }
    guard format==kCVPixelFormatType_OneComponent8 || format==kCVPixelFormatType_OneComponent32Float else {
        throw NSError(domain:"Mask",code:2,userInfo:[NSLocalizedDescriptionKey:"Unexpected pixel format \(format)"])
    }
    try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
    guard let provider=CGDataProvider(data:data as CFData),
          let cg=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:8,bytesPerRow:width,space:CGColorSpaceCreateDeviceGray(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent),
          let output=CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil) else {throw NSError(domain:"Mask",code:3)}
    CGImageDestinationAddImage(output,cg,[kCGImagePropertyOrientation:1] as CFDictionary)
    guard CGImageDestinationFinalize(output) else {throw NSError(domain:"Mask",code:4)}
    return ["width":width,"height":height,"pixel_format":Int(format)]
}

let manifestURL=URL(fileURLWithPath:CommandLine.arguments[1])
let resultsURL=URL(fileURLWithPath:CommandLine.arguments[2])
let rows=try JSONSerialization.jsonObject(with:Data(contentsOf:manifestURL)) as! [[String:String]]
var results:[[String:Any]]=[]
for (index,row) in rows.enumerated() {
    autoreleasepool {
        var record:[String:Any]=row
        do {
            guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:row["source"]!) as CFURL,nil),let image=CGImageSourceCreateImageAtIndex(source,0,nil) else {throw NSError(domain:"Mask",code:5)}
            let request=VNGenerateForegroundInstanceMaskRequest()
            request.preferBackgroundProcessing=true
            let handler=VNImageRequestHandler(cgImage:image,orientation:.up,options:[:])
            try handler.perform([request])
            guard let result=request.results?.first else {throw NSError(domain:"Mask",code:6)}
            record["source_width"]=image.width;record["source_height"]=image.height
            record["instance_ids"]=Array(result.allInstances)
            record["label_dimensions"]=try writeGray(result.instanceMask,to:URL(fileURLWithPath:row["instance_labels"]!))
            var details:[[String:Any]]=[]
            for id in result.allInstances {
                let instanceURL=URL(fileURLWithPath:row["instance_dir"]!).appendingPathComponent("instance_\(id).png")
                let mask=try result.generateScaledMaskForImage(forInstances:IndexSet(integer:id),from:handler)
                let dimensions=try writeGray(mask,to:instanceURL)
                details.append(["instance_id":id,"mask_path":instanceURL.path,"buffer":dimensions])
            }
            record["instances"]=details
            if !result.allInstances.isEmpty {
                let all=try result.generateScaledMaskForImage(forInstances:result.allInstances,from:handler)
                record["all_mask_dimensions"]=try writeGray(all,to:URL(fileURLWithPath:row["foreground_all"]!))
                record["status"]="generated"
            } else {record["status"]="no_foreground_instances"}
        } catch {record["status"]="error";record["error"]=String(describing:error)}
        results.append(record)
    }
    print("Foreground pilot \(index+1)/\(rows.count): \(row["image_name"]!)");fflush(stdout)
}
try JSONSerialization.data(withJSONObject:["stage":"prototype_only_not_approved_for_full290","method":"VNGenerateForegroundInstanceMaskRequest; all native-scaled per-instance mattes retained, no automatic selection", "images":results],options:[.prettyPrinted,.sortedKeys]).write(to:resultsURL,options:.atomic)
