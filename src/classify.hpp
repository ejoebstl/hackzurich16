struct MeasureInfo {
    int x, y, r1, r2;
    int matches, rejects, score; 
    int count, sum;
    double avg; 

    MeasureInfo() :
        x(0), y(0), r1(0), r2(0), matches(0), rejects(0), score(0), 
        count(0), sum(0), avg(0) {
    }
};

void ClassifyInit();
void ClassifyDeinit();
void detectAndDisplay(cv::Mat frame, MeasureInfo& leftSmall, MeasureInfo& leftLarge, MeasureInfo& rightSmall, MeasureInfo& rightLarge);



