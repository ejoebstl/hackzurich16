
inline cv::Rect Median(std::deque<cv::Rect> in) {
    std::sort(in.begin(), in.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.x < b.x;
            }); 

    int x = in[in.size() / 2].x;
    
    std::sort(in.begin(), in.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.y < b.y;
            }); 

    int y = in[in.size() / 2].y;
    
    std::sort(in.begin(), in.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.width < b.width;
            }); 

    int width = in[in.size() / 2].width;
    
    std::sort(in.begin(), in.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.height < b.height;
            }); 

    int height = in[in.size() / 2].height;

    return cv::Rect(x, y, width, height);
}

inline cv::Point Median(std::deque<cv::Point> in) {
    std::sort(in.begin(), in.end(), [](const cv::Point &a, const cv::Point &b) {
                return a.x < b.x;
            }); 

    int x = in[in.size() / 2].x;
    
    std::sort(in.begin(), in.end(), [](const cv::Point &a, const cv::Point &b) {
                return a.y < b.y;
            }); 

    int y = in[in.size() / 2].y;

    return cv::Point(x, y);
}  
