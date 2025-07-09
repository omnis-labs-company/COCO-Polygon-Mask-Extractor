#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <thread>

using namespace std;
namespace fs = std::filesystem;

// Convert polygons to binary mask
cv::Mat polygonsToMask(const vector<vector<cv::Point>>& polygons, const cv::Size& img_size) {
    cv::Mat mask = cv::Mat::zeros(img_size, CV_8UC1);
    for (const auto& poly : polygons) {
        vector<vector<cv::Point>> polylist{poly};
        cv::fillPoly(mask, polylist, cv::Scalar(255));
    }
    return mask;
}

// Extract polygons from segmentation field
vector<vector<cv::Point>> parsePolygons(const rapidjson::Value& seg) {
    vector<vector<cv::Point>> polygons;
    for (rapidjson::SizeType i = 0; i < seg.Size(); ++i) {
        const rapidjson::Value& arr = seg[i];
        vector<cv::Point> poly;
        for (rapidjson::SizeType j = 0; j < arr.Size(); j += 2) {
            int x = static_cast<int>(arr[j].GetDouble());
            int y = static_cast<int>(arr[j+1].GetDouble());
            poly.emplace_back(x, y);
        }
        polygons.push_back(poly);
    }
    return polygons;
}

// Build image_id to file_name map from JSON
unordered_map<int, string> buildImgIdToFileMap(const rapidjson::Document& coco) {
    unordered_map<int, string> id_to_file;
    const auto& imgs = coco["images"];
    for (rapidjson::SizeType i = 0; i < imgs.Size(); ++i) {
        const auto& img = imgs[i];
        int id = img["id"].GetInt();
        string fn = img["file_name"].GetString();
        id_to_file[id] = fn;
    }
    return id_to_file;
}

// Build category_id to name map from JSON
unordered_map<int, string> buildCategoryIdToNameMap(const rapidjson::Document& coco) {
    unordered_map<int, string> id_to_name;
    const auto& categories = coco["categories"];
    for (rapidjson::SizeType i = 0; i < categories.Size(); ++i) {
        const auto& c = categories[i];
        int cid = c["id"].GetInt();
        string name = c["name"].GetString();
        id_to_name[cid] = name;
    }
    return id_to_name;
}

// Load entire JSON file to string
string loadTextFile(const string& path) {
    ifstream in(path);
    string content((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    return content;
}

// Worker thread function for annotation chunk
void maskWriterThread(
    const vector<const rapidjson::Value*>& ann_chunk,
    const unordered_map<int, string>& category_map,
    const unordered_map<int, string>& imgid_map,
    const string& image_dir,
    const string& mask_dir)
{
    for (const auto* ann : ann_chunk) {
        int image_id = (*ann)["image_id"].GetInt();
        auto it = imgid_map.find(image_id);
        if (it == imgid_map.end()) {
            cerr << "[Thread] Image id not found for annotation id " << (*ann)["id"].GetInt() << endl;
            continue;
        }
        string img_path = (fs::path(image_dir) / it->second).string();

        cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            cerr << "[Thread] Image load failed: " << img_path << endl;
            continue;
        }

        if (!ann->HasMember("segmentation") || !(*ann)["segmentation"].IsArray() || (*ann)["segmentation"].Empty())
            continue;

        int category_id = (*ann)["category_id"].GetInt();
        string class_name = "unknown";
        if (category_map.find(category_id) != category_map.end()) {
            class_name = category_map.at(category_id);
        }
        int ann_id = (*ann)["id"].GetInt();

        // Make polygon mask
        auto polygons = parsePolygons((*ann)["segmentation"]);
        cv::Mat mask = polygonsToMask(polygons, image.size());

        // Get tight polygon bounding rect (ROI)
        cv::Rect tight_bbox = cv::boundingRect(polygons[0]);
        for (size_t j = 1; j < polygons.size(); ++j)
            tight_bbox |= cv::boundingRect(polygons[j]);
        tight_bbox = tight_bbox & cv::Rect(0, 0, image.cols, image.rows);
        if (tight_bbox.width <= 0 || tight_bbox.height <= 0) continue;

        cv::Mat cropped_img = image(tight_bbox);
        cv::Mat cropped_mask = mask(tight_bbox);

        // Output RGBA: polygon region has alpha=255, background alpha=0
        cv::Mat out_rgba(cropped_img.rows, cropped_img.cols, CV_8UC4, cv::Scalar(0,0,0,0));
        for (int y = 0; y < cropped_img.rows; ++y) {
            for (int x = 0; x < cropped_img.cols; ++x) {
                if (cropped_mask.at<uchar>(y, x) > 0) {
                    cv::Vec3b bgr = cropped_img.at<cv::Vec3b>(y, x);
                    out_rgba.at<cv::Vec4b>(y, x) = cv::Vec4b(bgr[0], bgr[1], bgr[2], 255);
                }
            }
        }

        string outname = class_name + "_" + to_string(ann_id) + ".png";
        fs::path outpath = fs::path(mask_dir) / outname;
        cv::imwrite(outpath.string(), out_rgba);
        cout << "[Thread] Saved: " << outpath << endl;
    }
}

int main() {
    string image_dir = "CVRG-Pano-20250709T025931Z-1-001\\CVRG-Pano\\all-rgb"; // image directory path
    string json_path = "COCO-Polygon-Mask-Extractor\\output.json"; // cocojson style annotations
    string mask_dir = "COCO-Polygon-Mask-Extractor\\masks"; // a directory path where we will save mask png files
    size_t num_threads = 8; // number of threads to use

    if (!fs::exists(mask_dir)) {
        fs::create_directories(mask_dir);
    }

    string json_str = loadTextFile(json_path);
    if (json_str.empty()) {
        cerr << "Cannot open json file: " << json_path << endl;
        return 1;
    }
    rapidjson::Document coco;
    coco.Parse(json_str.c_str());
    unordered_map<int, string> category_map = buildCategoryIdToNameMap(coco);
    unordered_map<int, string> imgid_map = buildImgIdToFileMap(coco);

    // Flatten all annotations into a vector of pointers for thread chunking!
    const auto& anns = coco["annotations"];
    vector<const rapidjson::Value*> all_anns;
    for (rapidjson::SizeType i = 0; i < anns.Size(); ++i)
        all_anns.push_back(&anns[i]);

    vector<vector<const rapidjson::Value*>> chunks(num_threads);
    for (size_t i = 0; i < all_anns.size(); ++i)
        chunks[i % num_threads].push_back(all_anns[i]);

    // Launch threads
    vector<thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            maskWriterThread,
            cref(chunks[i]), cref(category_map), cref(imgid_map),
            cref(image_dir), cref(mask_dir)
        );
    }
    for (auto& t : threads) t.join();

    cout << "Done: " << all_anns.size() << " objects extracted" << endl;
    return 0;
}