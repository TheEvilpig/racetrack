#include <iostream>
#include <vector>
#include <cstdint>
#include <omp.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace std;

struct Line {
    int16_t x1, y1, x2, y2;

    Line(int16_t _x1, int16_t _y1, int16_t _x2, int16_t _y2)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}

    static int orientation(int16_t ax, int16_t ay, int16_t bx, int16_t by, int16_t cx, int16_t cy) {
        int32_t val = (int32_t)(by - ay) * (cx - bx) - (int32_t)(bx - ax) * (cy - by);
        if (val > 0) return 1;
        if (val < 0) return 2;
        return 0;
    }

    static bool onSegment(int16_t px, int16_t py, int16_t qx, int16_t qy, int16_t rx, int16_t ry) {
        return (qx <= max(px, rx) && qx >= min(px, rx) &&
                qy <= max(py, ry) && qy >= min(py, ry));
    }

    bool intersects(const Line &other) const {
        int o1 = orientation(x1, y1, x2, y2, other.x1, other.y1);
        int o2 = orientation(x1, y1, x2, y2, other.x2, other.y2);
        int o3 = orientation(other.x1, other.y1, other.x2, other.y2, x1, y1);
        int o4 = orientation(other.x1, other.y1, other.x2, other.y2, x2, y2);

        if (o1 != o2 && o3 != o4) return true;
        if (o1 == 0 && onSegment(x1, y1, other.x1, other.y1, x2, y2)) return true;
        if (o2 == 0 && onSegment(x1, y1, other.x2, other.y2, x2, y2)) return true;
        if (o3 == 0 && onSegment(other.x1, other.y1, x1, y1, other.x2, other.y2)) return true;
        if (o4 == 0 && onSegment(other.x1, other.y1, x2, y2, other.x2, other.y2)) return true;
        return false;
    }
};

struct Accel {
    int16_t ax, ay;
};

struct Result {
    int16_t finalX, finalY;
    int16_t finalVx, finalVy;
    Accel sequence[5];

    //calculate thets :  angular position around track
    double calctheta(){

    }

    // Metric for ranking: distance squared from origin
    int metric() const { return finalX*finalX + finalY*finalY; }
};

// Generate all valid accelerations
vector<Accel> generateAccelerations() {
    vector<Accel> accels;
    for (int16_t ax = -10; ax <= 10; ax += 2) {
        for (int16_t ay = -10; ay <= 10; ay += 2) {

            if (ax == 0 && ay == 0) continue;
            if (abs(ax) == 2 && abs(ay) == 2) continue;
            if (abs(ax) == 2 && abs(ay) == 4) continue;
            if (abs(ax) == 4 && abs(ay) == 2) continue;
            if (abs(ax) == 0 && abs(ay) == 4) continue;
            if (abs(ax) == 4 && abs(ay) == 0) continue;
            if (abs(ax) == 2 && abs(ay) == 0) continue;
            if (abs(ax) == 0 && abs(ay) == 2) continue;
            
            if (ax*ax + ay*ay <= 100) accels.push_back({ax, ay});
        }
    }
    return accels;
}

// Progress monitor
void showProgress(size_t totalSequences, atomic<size_t>& sequences_done) {
    int secs = 0;
    while (sequences_done < totalSequences) {
        secs++;
        size_t done = sequences_done.load();
        int percent = (int)((done * 100) / totalSequences);
        cout << "\rProgress: [" 
             << string(percent / 2, '=') 
             << string(50 - percent / 2, ' ') 
             << "] " << percent << "%" 
             << ", " << done << "/" << totalSequences
             << ", " << secs/2 << "s"
             << flush;
        this_thread::sleep_for(chrono::milliseconds(500));
    }
    cout << "\rProgress: [" << string(50,'=') << "] 100%\n";
}

// Thread-safe top X simulation with 6 acceleration steps
vector<Result> simulateTopX(int16_t startX, int16_t startY,
                            int16_t startVx, int16_t startVy,
                            const vector<Line>& rLines,
                            int topX,
                            atomic<size_t>& sequences_done)
{
    vector<Accel> accels = generateAccelerations();
    size_t N = accels.size();
    size_t totalSequences = pow(N, 5);
    vector<Result> global_top;

    #pragma omp parallel
    {
        vector<Result> local_top;
        local_top.reserve(topX);

        #pragma omp for schedule(dynamic, 1000)
        for (size_t idx = 0; idx < totalSequences; idx++) {
            size_t i1 = idx / (N*N*N*N) % N;
            size_t i2 = idx / (N*N*N) % N;
            size_t i3 = idx / (N*N) % N;
            size_t i4 = idx / N % N;
            size_t i5 = idx % N;

            Accel seq[5] = { accels[i1], accels[i2], accels[i3], accels[i4], accels[i5] };

            int16_t x = startX, y = startY;
            int16_t vx = startVx, vy = startVy;
            bool crashed = false;

            for (auto a : seq) {
                int16_t avgVx = vx + a.ax / 2;
                int16_t avgVy = vy + a.ay / 2;
                int16_t nx = x + avgVx;
                int16_t ny = y + avgVy;
                Line move(x, y, nx, ny); 

                for (auto& wall : rLines)
                    if (move.intersects(wall)) { crashed = true; break; }

                if (crashed) break;

                x = nx; y = ny;
                vx += a.ax; vy += a.ay;
            }

            if (!crashed) {
                Result r = {x, y, vx, vy, {seq[0], seq[1], seq[2], seq[3], seq[4]}};

                if (local_top.size() < topX) {
                    local_top.push_back(r);
                    if (local_top.size() == topX)
                        sort(local_top.begin(), local_top.end(), [](const Result &a, const Result &b){ return a.metric() > b.metric(); });
                } else if (r.metric() > local_top.back().metric()) {
                    local_top.back() = r;
                    sort(local_top.begin(), local_top.end(), [](const Result &a, const Result &b){ return a.metric() > b.metric(); });
                }
            }
            sequences_done++;
        }

        #pragma omp critical
        {
            for (auto &r : local_top) {
                if (global_top.size() < topX) {
                    global_top.push_back(r);
                    if (global_top.size() == topX)
                        sort(global_top.begin(), global_top.end(), [](const Result &a, const Result &b){ return a.metric() > b.metric(); });
                } else if (r.metric() > global_top.back().metric()) {
                    global_top.back() = r;
                    sort(global_top.begin(), global_top.end(), [](const Result &a, const Result &b){ return a.metric() > b.metric(); });
                }
            }
        }
    }

    return global_top;
}

int main() {
    vector<pair<int16_t,int16_t>> iPoints = {
        {82, -80}, {82, 80}, {74, 94}, {60, 102}, {-60, 102}, {-74, 94}, {-82, 80}, {-82, -80}, {-74, -94}, {-60, -102}, {60, -102}, {74, -94}, {82, -80}
    };

    vector<pair<int16_t,int16_t>> oPoints = {
        {60, -60}, {60, 60}, {52, 72}, {40, 80}, {-40, 80}, {-52, 72}, {-60, 60}, {-60, -60}, {-52, -72}, {-40, -80}, {40, -80}, {52, -72}, {60, -60}
    };

    vector<Line> rLines;
    for (size_t i = 0; i + 1 < iPoints.size(); i++)
        rLines.emplace_back(iPoints[i].first, iPoints[i].second, iPoints[i+1].first, iPoints[i+1].second);
    for (size_t i = 0; i + 1 < oPoints.size(); i++)
        rLines.emplace_back(oPoints[i].first, oPoints[i].second, oPoints[i+1].first, oPoints[i+1].second);

    char cont = 'y';
    while (cont == 'y' || cont == 'Y') {
        int16_t startX, startY, startVx, startVy;
        cout << "Enter starting X Y: ";
        cin >> startX >> startY;
        cout << "Enter starting Vx Vy: ";
        cin >> startVx >> startVy;

        vector<Accel> accels = generateAccelerations();
        size_t totalSequences = pow(accels.size(), 5);
        atomic<size_t> sequences_done(0);

        thread monitor(showProgress, totalSequences, ref(sequences_done));

        int topX = 50; // store top 50 sequences globally
        vector<Result> topResults = simulateTopX(startX, startY, startVx, startVy, rLines, topX, sequences_done);

        monitor.join();

        cout << "Top " << topX << " sequences:\n";
        for (size_t i = 0; i < topResults.size(); i++) {
            cout << "Sequence " << i+1 << ": FinalPos=(" << topResults[i].finalX << "," << topResults[i].finalY 
                 << "), FinalVel=(" << topResults[i].finalVx << "," << topResults[i].finalVy << "), Accels=[";
            for (auto &a : topResults[i].sequence) cout << "(" << a.ax << "," << a.ay << ") ";
            cout << "]\n";
        }

        cout << "Do you want to continue? (y/n): ";
        cin >> cont;
    }
    cout << "Exiting program.\n";
}
