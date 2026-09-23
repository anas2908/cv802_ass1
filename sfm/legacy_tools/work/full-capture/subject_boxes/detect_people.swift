import Foundation
import Vision
import ImageIO
import CoreGraphics

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let data = try Data(contentsOf: inputURL)
let records = try JSONSerialization.jsonObject(with: data) as! [[String: String]]
var output: [[String: Any]] = []
for (i, record) in records.enumerated() {
    autoreleasepool {
        var result: [String: Any] = record
        let url = URL(fileURLWithPath: record["path"]!)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            result["error"] = "Could not decode image"
            output.append(result)
            return
        }
        let w = Double(image.width), h = Double(image.height)
        result["width"] = image.width
        result["height"] = image.height
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false
        do {
            let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
            try handler.perform([request])
            var candidates: [[String: Any]] = []
            for observation in request.results ?? [] {
                let b = observation.boundingBox
                let x0 = Double(b.minX) * w
                let y0 = (1.0 - Double(b.maxY)) * h
                let x1 = Double(b.maxX) * w
                let y1 = (1.0 - Double(b.minY)) * h
                let centerX = Double(b.midX)
                let centerY = Double(b.midY)
                let distance = sqrt(pow(centerX - 0.5, 2) + pow(centerY - 0.5, 2))
                let score = Double(b.width * b.height) * Double(observation.confidence) * max(0.2, 1.0 - 1.0 * distance)
                candidates.append(["bbox_xyxy": [x0,y0,x1,y1], "confidence": Double(observation.confidence), "score": score])
            }
            candidates.sort { ($0["score"] as! Double) > ($1["score"] as! Double) }
            result["candidates"] = candidates
            result["ambiguity"] = candidates.count > 1 && (candidates[1]["score"] as! Double) > (candidates[0]["score"] as! Double) * 0.65
            if let best = candidates.first {
                result["selected"] = best
                let bb = best["bbox_xyxy"] as! [Double]
                let bw = bb[2] - bb[0], bh = bb[3] - bb[1]
                let px = bw * (record["dataset"]! == "black_shirt_crutches" ? 0.24 : 0.10)
                let py = bh * 0.055
                result["padded_bbox_xyxy"] = [max(0,bb[0]-px),max(0,bb[1]-py),min(w,bb[2]+px),min(h,bb[3]+py)]
                result["status"] = "detected"
            } else { result["status"] = "not_detected" }
        } catch { result["status"] = "error"; result["error"] = String(describing: error) }
        output.append(result)
    }
    if (i+1) % 25 == 0 { print("Detected people in \(i+1)/\(records.count) images"); fflush(stdout) }
}
let json = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
try json.write(to: outputURL, options: .atomic)
print("Complete: \(output.count) records")
