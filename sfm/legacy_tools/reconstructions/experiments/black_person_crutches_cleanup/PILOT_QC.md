# Dark subject and crutches: segmentation pilot

This is a separate experiment. No source images, feature inputs, COLMAP models, application files or existing previews were changed. The full 290-image foreground-instance run was rejected after this pilot. Separately, person-only **body** masks are now being prepared with an explicit requirement for independent crutch protection; see `BODY_MASKS_README.md`.

Six upright source images were tested locally with Apple Vision person segmentation (`VNGeneratePersonSegmentationRequest`, accurate quality) and foreground-instance segmentation (`VNGenerateForegroundInstanceMaskRequest`). Both received orientation `.up`, matching the physically upright source images. The foreground results were generated at source resolution with `generateScaledMaskForImage`; the person matte was bilinearly resized to source resolution. The grayscale masks are confidence-like mattes, not calibrated probabilities. `pilot_overlay_00.jpg` through `pilot_overlay_05.jpg` show original pixels and analytical overlays.

## Finding

Neither automatic method reliably preserves the crutches. Every tested foreground request returned exactly one foreground instance; there is no separate detected crutch instance that can simply be added. A union of the two methods still misses exposed shaft sections in some views. Native-resolution masks align with the photographs, so the missing shafts are semantic segmentation failures rather than orientation or resize errors.

| Overlay | Source | Visual finding |
| --- | --- | --- |
| 00 | photos_iphone_11_26mm/0000_IMG_6653.jpg | Both methods keep the body but omit most long exposed shafts and their tips. A union cannot recover these sections. |
| 01 | photos_iphone_11_26mm/0076_IMG_8450.jpg | Foreground includes the image-right crutch more completely but drops much of the image-left shaft/tip; preservation is asymmetric. |
| 02 | photos_iphone_11_26mm/0112_IMG_8430.jpg | Parts overlapping the body survive, with some near-foot support, but this does not demonstrate reliable exposed-shaft preservation. |
| 03 | photos_iphone_11_26mm/0038_IMG_6673.jpg | Person matte includes the visible near crutch down to its tip; foreground loses much of the lower shaft/tip. Combining helps this image only. |
| 04 | video_IMG_6760/IMG_6760_t000417.jpg | Person matte retains more of the near crutch; foreground misses its lower exposed section and tip. |
| 05 | video_IMG_8288/IMG_8288_t007917.jpg | Both keep the body and upper overlapping sections while omitting exposed lower shafts; the frame itself also clips the lower scene. |

The numerical foreground-area differences in `pilot_qc_metrics.json` are small because crutches occupy few pixels. They are **not** an accuracy score and do not override the visual failures.

## Recommended bounded next experiment

Use person masks for body cleanup, and protect actual existing crutch points through a separate, image-grounded selection. Do not treat person masks as crutch silhouettes.

1. After the quality model finishes, choose roughly 8–12 clear, separated orbit photographs that show each physical crutch, including front and side directions. Work only on copies and write native-pixel polygons tightly around each visible shaft, foot, handle and underarm support. Keep the two physical crutch identities distinct; flag occlusion and cropped sections explicitly. Save original-plus-polygon overlays for review.
2. Project existing reconstructed points into these photographs using the finished calibrated cameras. Build a protection set from consistent crutch-polygon membership in at least three useful separated views, supplemented by actual feature-track observations in annotated regions when available. Ignore deliberately occluded or cropped sections rather than treating them as empty background. Apply the same distinct-view, reprojection-error and triangulation-angle quality checks used for the comparison. Keep rejected/ambiguous candidates and counts in the audit.
3. Visually inspect retained candidates overlaid onto all annotated views before choosing the final protection gate. Narrow image corridors can admit unrelated room points along a ray, so a single image or a broad bounding box is insufficient.
4. In a new output only, retain points that pass body-mask filtering OR the verified crutch-protection set. Retain original 3D coordinates and colors. This removes points; it adds no geometry and cannot recover shaft surfaces absent from SfM.

This is preferable to broad mask dilation or generic foreground-instance selection: those operations do not establish which thin structures are the real crutches. If the 8–12-view annotations cannot separate reliable crutch points, report that limitation and preserve the existing preview rather than presenting a cut-off result as crutch-preserving.

## Files and provenance

- `pilot_manifest.json`: six source paths and output mapping.
- `pilot_results.json`: foreground API results, instance IDs and image dimensions.
- `person_metadata/`: person-segmentation raw output and orientation information.
- `foreground_instances/`, `foreground_all/`, `person_only/`: experimental mattes only.
- `scripts/`: reproducible local generation and analytical QC scripts.

Apple API references:
- https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest
- https://developer.apple.com/documentation/vision/vngenerateforegroundinstancemaskrequest
- https://developer.apple.com/documentation/vision/vninstancemaskobservation/generatescaledmaskforimage%28forinstances%3Afrom%3A%29
