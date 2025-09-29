#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <fstream>
#include <functional>

using namespace std::chrono;

using namespace std;
using namespace cv;
const char* window_name = "Panorama Project";
const int MINIMUM_MATCHES = 5;

static const int MAX_CANVAS_WIDTH = 8000;
static const int MAX_CANVAS_HEIGHT = 8000;

void show_image(Mat img, String title = "Title", float scale = 0.25) {
    Mat small_img;
    resize(img, small_img, Size(), scale, scale, INTER_AREA);
    imshow(title, small_img);
    destroyWindow(title);
}

tuple<Mat, vector<KeyPoint>, Mat> run_sift(const Mat& img, const Mat& greyscale) {
    auto sift = cv::SIFT::create(0);
    Mat descriptors;
    vector<KeyPoint> key_points;
    sift->detectAndCompute(greyscale, noArray(), key_points, descriptors);

    Mat result;
    drawKeypoints(img, key_points, result, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return { result, key_points, descriptors };
}

tuple<Mat, vector<KeyPoint>, Mat> run_orb(const Mat& img, const Mat& greyscale) {
    auto orb = cv::ORB::create(10000);
    Mat descriptors;
    vector<KeyPoint> key_points;
    orb->detectAndCompute(greyscale, noArray(), key_points, descriptors);

    Mat result;
    drawKeypoints(img, key_points, result, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return { result, key_points, descriptors };
}

void iterate_images(function<void(string)> fn) {
    for (int i = 1; i < 4; i++) {
        for (int j = 1; j < 4; j++) {
            string filename = to_string(i) + to_string(j) + ".jpg";
            fn(filename);
        }
    }
}

void make_histogram(vector<cv::DMatch> data, string dir) {
    vector<float> distances;
    for (cv::DMatch d : data) {
        distances.push_back(d.distance);
    }
    float max, min;
    min = *min_element(distances.begin(), distances.end());
    max = *max_element(distances.begin(), distances.end());
    int bins = 20;
    Mat distance_matrix(distances.size(), 1, CV_32F, distances.data());
    Mat hist;
    float range[] = { min, max };
    const float* histRange = { range };
    calcHist(&distance_matrix, 1, 0, Mat(), hist, 1, &bins, &histRange, true, false);

    int width = 700;
    int height = 500;
    int left_margin = 80;
    int bottom_margin = 80;
    int top_margin = 80;
    int right_margin = 20;

    Mat histogram(height, width, CV_8UC3, Scalar(255, 255, 255));

    double maxFreq;
    minMaxLoc(hist, 0, &maxFreq);

    int plot_height = height - bottom_margin - top_margin;
    int plot_width = width - left_margin - right_margin;

    int bar_width = plot_width / bins;
    for (int bin = 0; bin < bins; bin++) {
        float freq = hist.at<float>(bin);
        int bar_height = (int)(freq / maxFreq * plot_height);
        rectangle(histogram,
            Point(left_margin + bin * bar_width, height - bottom_margin - bar_height),
            Point(left_margin + (bin + 1) * bar_width - 1, height - bottom_margin),
            Scalar(200, 100, 0),
            FILLED);
    }

    line(histogram, Point(left_margin, height - bottom_margin),
        Point(width - right_margin, height - bottom_margin), Scalar(0, 0, 0), 2);
    line(histogram, Point(left_margin, height - bottom_margin),
        Point(left_margin, top_margin), Scalar(0, 0, 0), 2);

    for (int i = 0; i <= 5; i++) {
        float value = min + (max - min) * i / 5.0f;
        int x_pos = left_margin + plot_width * i / 5;
        line(histogram, Point(x_pos, height - bottom_margin),
            Point(x_pos, height - bottom_margin + 5), Scalar(0, 0, 0), 2);
        putText(histogram, cv::format("%.2f", value),
            Point(x_pos - 20, height - bottom_margin + 25),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    for (int i = 0; i <= 5; i++) {
        int freq_value = (int)(maxFreq * i / 5.0);
        int y_pos = height - bottom_margin - plot_height * i / 5;
        line(histogram, Point(left_margin - 5, y_pos),
            Point(left_margin, y_pos), Scalar(0, 0, 0), 2);
        putText(histogram, to_string(freq_value),
            Point(left_margin - 50, y_pos + 5),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    putText(histogram, "Match Distance Histogram",
        Point(width / 2 - 150, 40),
        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    putText(histogram, "Match Distance",
        Point(width / 2 - 70, height - 20),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    putText(histogram, "Frequency",
        Point(left_margin - 15, top_margin - 20),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    putText(histogram, "Matches: " + to_string(data.size()),
        Point(left_margin, height - 5),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    //show_image(histogram, "Histogram", 1);
    imwrite(dir, histogram);
}

void add_line_to_file(string filename, string line, bool silent = false) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << line;
        file.close();
        if (!silent) {
            std::cout << "    " << line;
        }
    }
    else {
        std::cout << "Unable to open file\n";
    }
}

void prepare_data_dir(string dir, string header = "") {
    filesystem::create_directories(dir);
    if (!header.empty()) {
        ofstream data_file(dir + "results.txt");
        if (data_file.is_open()) {
            data_file << header;
            data_file.close();
        }
    }
}

struct homography_input {
    vector<KeyPoint> key_points_1;
    vector<KeyPoint> key_points_2;
    vector<cv::DMatch> matches;
    Mat img1;
    Mat img2;
};

struct homography_result {
    Mat homography_matrix;
    Mat img1;
    Mat img2;
};
vector<vector<homography_result>> estimate_homography(vector<vector<homography_input>> homography_input, string dir, float reprojection_thres) {
    vector<vector<homography_result>> results;
    for (int i = 0; i < homography_input.size(); i++) {
        vector<homography_result> set_results;
        for (int j = 0; j < homography_input[i].size(); j++) {
            homography_result result;
            int inlier_count = 0;
            int outlier_count = 0;
            microseconds duration = microseconds(0);

            if (homography_input[i][j].matches.size() < MINIMUM_MATCHES) {
                cout << "Not enough matches found to properly est. homography (found "
                    << homography_input[i][j].matches.size() << ", need " << MINIMUM_MATCHES << ")" << endl;

                add_line_to_file(dir + "results.txt",
                    to_string(reprojection_thres) + " " + to_string(inlier_count) + " " +
                    to_string(homography_input[i][j].matches.size()) + " " +
                    to_string(duration.count()) + "\n");

                set_results.push_back(result);
                continue;
            }

            cv::Mat img2_with_box = homography_input[i][j].img2.clone();
            vector<cv::Point2f> src_pts, dst_pts;
            for (int k = 0; k < homography_input[i][j].matches.size(); k++) {
                const auto& match = homography_input[i][j].matches[k];
                src_pts.push_back(homography_input[i][j].key_points_1[match.queryIdx].pt);
                dst_pts.push_back(homography_input[i][j].key_points_2[match.trainIdx].pt);
            }

            auto start = high_resolution_clock::now();
            cv::Mat mask;
            cv::Mat M = cv::findHomography(src_pts, dst_pts,
                cv::RANSAC, reprojection_thres, mask);
            auto stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);

            if (M.empty()) {
                outlier_count = homography_input[i][j].matches.size();

                add_line_to_file(dir + "results.txt",
                    to_string(reprojection_thres) + " " + to_string(inlier_count) + " " +
                    to_string(outlier_count) + " " +
                    to_string(duration.count()) + "\n");

                set_results.push_back(result);
                continue;
            }

            for (int m = 0; m < mask.rows; m++) {
                if (mask.at<uchar>(m, 0) == 1) {
                    inlier_count++;
                }
            }

            outlier_count = homography_input[i][j].matches.size() - inlier_count;
            result.homography_matrix = M;
			result.img1 = homography_input[i][j].img1;
			result.img2 = homography_input[i][j].img2;

            Mat warped;
            cv::warpPerspective(homography_input[i][j].img1, warped, M,
                homography_input[i][j].img2.size());

            Mat blended;
            cv::addWeighted(warped, 1, homography_input[i][j].img2, 1, 0, blended);

            vector<cv::Point2f> corners(4);
            corners[0] = Point2f(0, 0);
            corners[1] = Point2f(homography_input[i][j].img1.cols - 1, 0);
            corners[2] = Point2f(homography_input[i][j].img1.cols - 1, homography_input[i][j].img1.rows - 1);
            corners[3] = Point2f(0, homography_input[i][j].img1.rows - 1);

            vector<cv::Point2f> transformed_corners;
            cv::perspectiveTransform(corners, transformed_corners, M);

            vector<Point> poly_corners;
            for (const auto& pt : transformed_corners) {
                poly_corners.push_back(cv::Point((int) (pt.x),
                    (int) (pt.y)));
            }

            polylines(blended, poly_corners, true,
                cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

            imwrite(dir + "homography/" + to_string(reprojection_thres) + "/" +
                to_string(i+1) + to_string(j+1) + ".png", blended);

            add_line_to_file(dir + "results.txt",
                to_string(reprojection_thres) + " " + to_string(inlier_count) + " " +
                to_string(outlier_count) + " " +
                to_string(duration.count()) + "\n");

            set_results.push_back(result);
        }
        results.push_back(set_results);
    }
    return results;
}

struct coords {
    float x;
    float y;
};

tuple<Mat, float, float> get_panorama_canvas(Mat img, Mat H) {
    vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f(img.cols - 1, 0);
    corners[2] = Point2f(img.cols - 1, img.rows - 1);
    corners[3] = Point2f(0, img.rows - 1);

    vector<Point2f> transformed_corners;
    perspectiveTransform(corners, transformed_corners, H);

    float xmin = transformed_corners[0].x;
    float ymin = transformed_corners[0].y;
    float xmax = transformed_corners[0].x;
    float ymax = transformed_corners[0].y;

    for (size_t i = 1; i < transformed_corners.size(); i++) {
        xmin = std::min(xmin, transformed_corners[i].x);
        ymin = std::min(ymin, transformed_corners[i].y);
        xmax = std::max(xmax, transformed_corners[i].x);
        ymax = std::max(ymax, transformed_corners[i].y);
    }

    Mat panorama_canvas = Mat::zeros(
        Size(min((int)(xmax - xmin), MAX_CANVAS_WIDTH), min((int)(ymax - ymin), MAX_CANVAS_HEIGHT)),
        img.type()
    );
    return tuple(panorama_canvas, xmin, ymin);
}

Mat stitch_panorama_feathered(const vector<homography_result> results) {
    Mat H12 = results[0].homography_matrix;
    Mat H32 = results[1].homography_matrix;
    Mat img1 = results[0].img1;
    Mat img2 = results[0].img2;
    Mat img3 = results[1].img2;

    CV_Assert(img1.type() == CV_8UC4);
    CV_Assert(img2.type() == CV_8UC4);
    CV_Assert(img3.type() == CV_8UC4);

    auto [canvas1, xmin1, ymin1] = get_panorama_canvas(img1, H12);
    float xmax1 = xmin1 + canvas1.cols;
    float ymax1 = ymin1 + canvas1.rows;
    Mat H32_inv = H32.inv();
    auto [canvas3, xmin3, ymin3] = get_panorama_canvas(img3, H32_inv);
    float xmax3 = xmin3 + canvas3.cols;
    float ymax3 = ymin3 + canvas3.rows;
    float final_xmin = std::min({ xmin1, xmin3, 0.0f });
    float final_ymin = std::min({ ymin1, ymin3, 0.0f });
    float final_xmax = std::max({ xmax1, xmax3, (float)img2.cols });
    float final_ymax = std::max({ ymax1, ymax3, (float)img2.rows });
    int canvas_width = (int) (std::ceil(final_xmax - final_xmin));
    int canvas_height = (int) (std::ceil(final_ymax - final_ymin));

    Mat panorama_canvas = Mat::zeros(Size(canvas_width, canvas_height), CV_8UC4);
    Mat alpha_accumulator = Mat::zeros(Size(canvas_width, canvas_height), CV_32FC1);
    Mat color_accumulator = Mat::zeros(Size(canvas_width, canvas_height), CV_32FC3);

    Mat translation_matrix = (Mat_<double>(3, 3) <<
        1, 0, -final_xmin,
        0, 1, -final_ymin,
        0, 0, 1);

    Mat warped1 = Mat::zeros(panorama_canvas.size(), CV_8UC4);
    cv::warpPerspective(img1, warped1, translation_matrix * H12, warped1.size());

    Mat warped3 = Mat::zeros(panorama_canvas.size(), CV_8UC4);
    cv::warpPerspective(img3, warped3, translation_matrix * H32_inv, warped3.size());

    for (int y = 0; y < canvas_height; ++y) {
        for (int x = 0; x < canvas_width; ++x) {
            cv::Vec3f color_sum(0, 0, 0);
            float alpha_sum = 0.0f;

            Vec4b pixel1 = warped1.at<Vec4b>(y, x);
            float alpha1 = pixel1[3] / 255.0f;
            if (alpha1 > 0) {
                color_sum[0] += pixel1[0] * alpha1;
                color_sum[1] += pixel1[1] * alpha1;
                color_sum[2] += pixel1[2] * alpha1;
                alpha_sum += alpha1;
            }

            Vec4b pixel3 = warped3.at<Vec4b>(y, x);
            float alpha3 = pixel3[3] / 255.0f;
            if (alpha3 > 0) {
                color_sum[0] += pixel3[0] * alpha3;
                color_sum[1] += pixel3[1] * alpha3;
                color_sum[2] += pixel3[2] * alpha3;
                alpha_sum += alpha3;
            }

            int img2_x = x - (int) (round(-final_xmin));
            int img2_y = y - (int) (round(-final_ymin));
            if (img2_x >= 0 && img2_x < img2.cols && img2_y >= 0 && img2_y < img2.rows) {
                Vec4b pixel2 = img2.at<Vec4b>(img2_y, img2_x);
                float alpha2 = pixel2[3] / 255.0f;
                if (alpha2 > 0) {
                    color_sum[0] += pixel2[0] * alpha2;
                    color_sum[1] += pixel2[1] * alpha2;
                    color_sum[2] += pixel2[2] * alpha2;
                    alpha_sum += alpha2;
                }
            }

            if (alpha_sum > 0) {
                panorama_canvas.at<Vec4b>(y, x) = Vec4b(
                    (uchar) (color_sum[0] / alpha_sum),
                    (uchar) (color_sum[1] / alpha_sum),
                    (uchar) (color_sum[2] / alpha_sum),
                    255
                );
            }
        }
    }

    Mat result;
    cvtColor(panorama_canvas, result, COLOR_BGRA2BGR);
    return result;
}

Mat stitch_panorama(const vector<homography_result> results) {
    Mat H12 = results[0].homography_matrix;
    Mat H32 = results[1].homography_matrix;

    Mat img1 = results[0].img1;
    Mat img2 = results[0].img2;
    Mat img3 = results[1].img2;

    auto [canvas1, xmin1, ymin1] = get_panorama_canvas(img1, H12);
    float xmax1 = xmin1 + canvas1.cols;
    float ymax1 = ymin1 + canvas1.rows;

    Mat H32_inv = H32.inv();
    auto [canvas3, xmin3, ymin3] = get_panorama_canvas(img3, H32_inv);
    float xmax3 = xmin3 + canvas3.cols;
    float ymax3 = ymin3 + canvas3.rows;

    float final_xmin = std::min({ xmin1, xmin3, 0.0f });
    float final_ymin = std::min({ ymin1, ymin3, 0.0f });
    float final_xmax = std::max({ xmax1, xmax3, (float)img2.cols });
    float final_ymax = std::max({ ymax1, ymax3, (float)img2.rows });

    int canvas_width = (int) (std::ceil(final_xmax - final_xmin));
    int canvas_height = (int) (std::ceil(final_ymax - final_ymin));

    Mat panorama_canvas = Mat::zeros(Size(canvas_width, canvas_height), img1.type());

    Mat translation_matrix = (Mat_<double>(3, 3) <<
        1, 0, -final_xmin,
        0, 1, -final_ymin,
        0, 0, 1);

    cv::warpPerspective(img1, panorama_canvas, translation_matrix * H12, panorama_canvas.size());

    cv::warpPerspective(img3, panorama_canvas, translation_matrix * H32_inv, panorama_canvas.size(),
        INTER_LINEAR, BORDER_TRANSPARENT);

    int roi_x = (int) (round(-final_xmin));
    int roi_y = (int) (round(-final_ymin));
    Mat roi = panorama_canvas(Rect(roi_x, roi_y, img2.cols, img2.rows));
    img2.copyTo(roi);

    return panorama_canvas;
}

Mat add_gradient_alpha(const Mat M, const Mat gradient) {
    CV_Assert(M.type() == CV_8UC3);
    CV_Assert(gradient.type() == CV_32F);
    CV_Assert(M.size() == gradient.size());

    cv::Mat result(M.rows, M.cols, CV_8UC4);

    for (int y = 0; y < M.rows; ++y) {
        for (int x = 0; x < M.cols; ++x) {
            cv::Vec3b bgr = M.at<cv::Vec3b>(y, x);
            uchar alpha = (uchar)(gradient.at<float>(y, x) * 255.0f + 0.5f);
            result.at<Vec4b>(y, x) = Vec4b(bgr[0], bgr[1], bgr[2], alpha);
        }
    }
    return result;
}

Mat make_gradient_mask(int width, int height, int blend_width, bool fade_left, bool fade_right) {
    Mat mask = Mat::ones(height, width, CV_32F);
    if (fade_left) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < min(blend_width, width); x++) {
                mask.at<float>(y, x) = (float)x / blend_width;
            }
        }
    }
    if (fade_right) {
        for (int y = 0; y < height; y++) {
            for (int x = max(0, width - blend_width); x < width; x++) {
                mask.at<float>(y, x) = 1.0f - (float)(x - (width - blend_width)) / blend_width;
            }
        }
    }
    return mask;
}

int main(int argc, char** argv) {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    // Delete old data
    filesystem::remove_all("../../../SIFT/");
    filesystem::remove_all("../../../ORB/");

    string main_dir = "../../../images/";
    string sift_dir = "../../../SIFT/";
    string orb_dir = "../../../ORB/";

    prepare_data_dir(sift_dir, "SIFT Feature Detection Results\nFilename KeyPoints\n");
    prepare_data_dir(sift_dir + "key points/");
    prepare_data_dir(orb_dir, "ORB Feature Detection Results\nFilename KeyPoints\n");
    prepare_data_dir(orb_dir + "key points/");

    vector<Mat> images;
    vector < pair<vector<KeyPoint>, Mat>> sift_detections, orb_detections;
    cout << "Running SIFT / ORB feature detection on image set" << endl;
    cout << "    Filename KeyPoints" << endl;
    iterate_images([&](string filename) {
        Mat img = imread(main_dir + filename);
        images.push_back(img);
        Mat greyscale; cvtColor(img, greyscale, COLOR_BGR2GRAY);

        // SIFT features
        auto [sift_img, sift_key_points, sift_descriptors] =
            run_sift(img, greyscale);
        add_line_to_file(sift_dir + "results.txt",
            filename + " " + to_string(sift_key_points.size()) + "\n");
        imwrite(sift_dir + "key points/" + filename, sift_img);
        sift_detections.push_back(pair(sift_key_points, sift_descriptors));

        // ORB features
        auto [orb_img, orb_key_points, orb_descriptors] =
            run_orb(img, greyscale);
        add_line_to_file(orb_dir + "results.txt",
            filename + " " + to_string(orb_key_points.size()) + "\n");
        imwrite(orb_dir + "key points/" + filename, orb_img);
        orb_detections.push_back(pair(orb_key_points, orb_descriptors));
        });

    add_line_to_file(sift_dir + "results.txt", "\nSIFT Feature Matching results\nImages Matching Time (ms):\n", true);
    add_line_to_file(orb_dir + "results.txt", "\nORB Feature Matching results\nImages Feature Matching Time (ms):\n", true);

    vector<vector<vector<cv::DMatch>>> sift_matches, orb_matches;
    vector<vector<homography_input>> sift_homography_data, orb_homography_data;
    sift_homography_data.resize(3, vector<homography_input>(2));
    orb_homography_data.resize(3, vector<homography_input>(2));

    // Create matchers
    cv::FlannBasedMatcher flann_sift_matcher;
    cv::BFMatcher bf_orb_matcher(cv::NORM_HAMMING);

    printf("\nMatching features\n");
    printf("    Images Matching Time(ms):\n");
    for (int set = 0; set < 3; set++) {
        for (int img = 0; img < 2; img++) {
            std::ostringstream line;
            line << set + 1 << (img + 1) << ".jpg and "
                << set + 1 << (img + 2) << ".jpg ";

            int imgIndex = set * 3 + img;

            auto [sift_kp1, sift_desc1] = sift_detections[imgIndex];
            auto [sift_kp2, sift_desc2] = sift_detections[imgIndex + 1];

            vector<vector<cv::DMatch>> sift_knn_matches;
            auto start = high_resolution_clock::now();
            flann_sift_matcher.knnMatch(sift_desc1, sift_desc2, sift_knn_matches, 2);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            add_line_to_file(sift_dir + "results.txt",
                line.str() + to_string(duration.count()) + "\n");

            vector<cv::DMatch> sift_feature_matches;
            float ratio_thresh = 0.7f;
            for (size_t i = 0; i < sift_knn_matches.size(); i++) {
                if (sift_knn_matches[i].size() >= 2 &&
                    sift_knn_matches[i][0].distance < ratio_thresh * sift_knn_matches[i][1].distance) {
                    sift_feature_matches.push_back(sift_knn_matches[i][0]);
                }
            }

            sift_homography_data[set][img] = {
                sift_kp1,
                sift_kp2,
                sift_feature_matches,
                images[imgIndex],
                images[imgIndex + 1]
            };

            auto [orb_kp1, orb_desc1] = orb_detections[imgIndex];
            auto [orb_kp2, orb_desc2] = orb_detections[imgIndex + 1];

            vector<vector<cv::DMatch>> orb_knn_matches;

            start = high_resolution_clock::now();
            bf_orb_matcher.knnMatch(orb_desc1, orb_desc2, orb_knn_matches, 2);

            stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);
            add_line_to_file(orb_dir + "results.txt",
                line.str() + to_string(duration.count()) + "\n");

            vector<cv::DMatch> orb_feature_matches;
            for (size_t i = 0; i < orb_knn_matches.size(); i++) {
                if (orb_knn_matches[i].size() >= 2 &&
                    orb_knn_matches[i][0].distance < ratio_thresh * orb_knn_matches[i][1].distance) {
                    orb_feature_matches.push_back(orb_knn_matches[i][0]);
                }
            }

            orb_homography_data[set][img] = {
                orb_kp1,
                orb_kp2,
                orb_feature_matches,
                images[imgIndex],
                images[imgIndex + 1]
            };
        }
    }

    printf("\nMaking histograms\n");
    prepare_data_dir(sift_dir + "histograms", "");
    prepare_data_dir(orb_dir + "histograms");
    for (int set = 0; set < 3; set++) {
        for (int img = 0; img < 2; img++) {
            string histogram_subdir = "histograms/" + to_string(set + 1) + to_string(img + 1) + ".png";
            make_histogram(sift_homography_data[set][img].matches, sift_dir + histogram_subdir);
            make_histogram(orb_homography_data[set][img].matches, orb_dir + histogram_subdir);
        }
    }

    printf("\nEstimating homography\n");

    add_line_to_file(sift_dir + "results.txt", "\nReproj. Thres. Inliers Outliers Time (ms)\n", true);
    add_line_to_file(orb_dir + "results.txt", "\nReproj. Thres. Inliers Outliers Time (ms)\n", true);

    vector<vector<homography_result>> sift_homography_results, orb_homography_results;

    for (float threshold : vector<float>({ 0.6, 1, 3, 5, 10 })) {
        printf("    Reprojection threshold: %.1f\n", threshold);
        prepare_data_dir(sift_dir + "homography/" + to_string(threshold) + "/");

        sift_homography_results = estimate_homography(sift_homography_data, sift_dir, threshold);

        prepare_data_dir(orb_dir + "homography/" + to_string(threshold) + "/");
        orb_homography_results = estimate_homography(orb_homography_data, orb_dir, threshold);
    }

	cout << "\nStitching panoramas\n";

    prepare_data_dir(sift_dir + "panoramas/simple/");
    prepare_data_dir(sift_dir + "panoramas/feathered/");
    prepare_data_dir(orb_dir + "panoramas/simple/");
    prepare_data_dir(orb_dir + "panoramas/feathered/");

    for (int set = 0; set < 3; set++) {
		cout << "    Image set " << (set + 1) << endl;
        string triple_name = to_string(set + 1);

        // SIFT triple panorama (simple)
        auto sift_panorama_simple = stitch_panorama(sift_homography_results[set]);
        imwrite(sift_dir + "panoramas/simple/" + triple_name + ".png",
            sift_panorama_simple);

        // ORB triple panorama (simple)
        auto orb_panorama_simple = stitch_panorama(orb_homography_results[set]);
        imwrite(orb_dir + "panoramas/simple/" + triple_name + ".png",
            orb_panorama_simple);

        // Apply gradient masks to each image
        sift_homography_results[set][0].img1 = add_gradient_alpha(
            sift_homography_results[set][0].img1,
            make_gradient_mask(sift_homography_results[set][0].img1.cols,
                sift_homography_results[set][0].img1.rows, 100, false, true)
        );

        sift_homography_results[set][0].img2 = add_gradient_alpha(
            sift_homography_results[set][0].img2,
            make_gradient_mask(sift_homography_results[set][0].img2.cols,
                sift_homography_results[set][0].img2.rows, 100, true, true)
        );

        sift_homography_results[set][1].img2 = add_gradient_alpha(
            sift_homography_results[set][1].img2,
            make_gradient_mask(sift_homography_results[set][1].img2.cols,
                sift_homography_results[set][1].img2.rows, 100, true, false)
        );

        // SIFT triple panorama (feathered)
        auto sift_panorama_feathered = stitch_panorama_feathered(sift_homography_results[set]);
        imwrite(sift_dir + "panoramas/feathered/" + triple_name + ".png",
            sift_panorama_feathered);

        // Apply gradient masks to each image
        orb_homography_results[set][0].img1 = add_gradient_alpha(
            orb_homography_results[set][0].img1,
            make_gradient_mask(orb_homography_results[set][0].img1.cols,
                orb_homography_results[set][0].img1.rows, 100, false, true)
        );

        orb_homography_results[set][0].img2 = add_gradient_alpha(
            orb_homography_results[set][0].img2,
            make_gradient_mask(orb_homography_results[set][0].img2.cols,
                orb_homography_results[set][0].img2.rows, 100, true, true)
        );

        orb_homography_results[set][1].img2 = add_gradient_alpha(
            orb_homography_results[set][1].img2,
            make_gradient_mask(orb_homography_results[set][1].img2.cols,
                orb_homography_results[set][1].img2.rows, 100, true, false)
        );

        // ORB triple panorama (feathered)
        auto orb_panorama_feathered = stitch_panorama_feathered(orb_homography_results[set]);
        imwrite(orb_dir + "panoramas/feathered/" + triple_name + ".png",
            orb_panorama_feathered);
    }
}