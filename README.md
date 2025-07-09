# COCO Polygon Mask Extractor (C++/OpenCV/RapidJSON/Multithread)
Extract each object instance from COCO-format polygon annotations as transparent PNG mask images — 

![Concept](https://github.com/omnis-labs-company/COCO-Polygon-Mask-Extractor)

**fast & parallel!**

## Features

- **Supports standard COCO JSON format**: parses annotations, images, categories.
- **Multi-threaded**: automatic multi-thread  parallelization for maximum speed.
- **Polygon-grounded PNG Export**: objects are extracted as `class_annotationid.png` with transparent backgrounds (true alpha).

---

## Example Workflow

Suppose you have:
- A COCO dataset JSON at `output.json`. Please check https://github.com/omnis-labs-company/raster-mask-to-cocojson if you want to generate sample output.json file.
- Images in a directory (e.g. `all-rgb`)
- Want to save all mask PNGs to a directory (e.g. `masks`)

### Instruction:

1. Edit the source code paths at the top for:
    - `image_dir` (directory with the images)
    - `json_path` (COCO annotation JSON)
    - `mask_dir` (where you want output PNGs)
    - `num_threads` (Increase the number if you have an expensive CPU)

2. Build and run the program.  
   Each mask PNG will be named like:  
   ```
   <class>_<annotationid>.png
   ```
   For example: `car_142.png`

---

## Requirements

- C++17 or higher
- OpenCV (recommended 4.x)
- RapidJSON ([GitHub link](https://github.com/Tencent/rapidjson))
- A compiler with threads and filesystem support (GCC 7+/Clang 7+/MSVC 2017+)

---

## Input/Output Example

**Input:**  
- output.json: COCO instance segmentation annotation file  
- all-rgb/: contains 000001.jpg, 000002.jpg, ...  
- Each annotation contains an "image_id" and connects to an image in the "images" array.

**Output:**  
- masks/car_1000123.png
- masks/person_1000321.png
- ...  
  (One PNG per annotation, transparent background, polygon tight crop)

---

## Customization

- To change the number of worker threads, adjust num_threads = 8; in main.
- Output mask naming format can be changed in the relevant code line.

---

## Troubleshooting

- If a source image does not exist, or the annotation is malformed, that mask is skipped (continue).
- Ensure that all images listed in COCO "images" exist in your image_dir.

---

## License

MIT (or your chosen license)

---

## Credits

- OpenCV: computer vision toolbox
- RapidJSON: lightning fast JSON parser