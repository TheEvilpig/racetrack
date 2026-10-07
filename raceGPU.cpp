#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>

struct Accel { int16_t ax, ay; };
struct Result {
    int16_t x, y, vx, vy;
    Accel seq[6];
    int metric() const { return x*x + y*y; }
};

std::vector<Accel> generateAccelerations() {
    std::vector<Accel> accels;
    for(int16_t ax=-10; ax<=10; ax+=2)
        for(int16_t ay=-10; ay<=10; ay+=2)
            if(abs(ax)<8 && abs(ay)<8) continue;
            else accels.push_back({ax, ay});
    return accels;
}

void mergeTopX(std::vector<Result>& globalTop, std::vector<Result>& blockTop, size_t topX) {
    globalTop.insert(globalTop.end(), blockTop.begin(), blockTop.end());
    std::sort(globalTop.begin(), globalTop.end(),
              [](const Result& a, const Result& b){ return a.metric() > b.metric(); });
    if(globalTop.size() > topX) globalTop.resize(topX);
}

// CPU thread to show progress
void showProgress(std::atomic<size_t>& chunksDone, size_t totalChunks) {
    while(chunksDone < totalChunks) {
        int percent = (int)((chunksDone.load()*100)/totalChunks);
        std::cout << "\rProgress: [" 
                  << std::string(percent/2,'=') 
                  << std::string(50-percent/2,' ') 
                  << "] " << percent << "%" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "\rProgress: [" << std::string(50,'=') << "] 100%\n";
}

int main() {
    auto accels = generateAccelerations();
    size_t N = accels.size();
    size_t totalSequences = std::pow(N, 6);
    size_t topX = 50;
    std::vector<Result> globalTop;

    // OpenCL setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer accelBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(Accel)*accels.size(), accels.data());

    std::string kernelCode = R"CLC(
    typedef short int16_t;
    typedef struct { int16_t ax, ay; } Accel;
    typedef struct { int16_t x,y,vx,vy; Accel seq[6]; } Result;

    __kernel void simulate(__global const Accel* accels, const int N,
                           __global Result* results, const int offset) {
        int gid = get_global_id(0);
        int idx = gid + offset;

        int i1 = idx / (N*N*N*N*N) % N;
        int i2 = idx / (N*N*N*N) % N;
        int i3 = idx / (N*N*N) % N;
        int i4 = idx / (N*N) % N;
        int i5 = idx / N % N;
        int i6 = idx % N;

        Accel seq[6] = {accels[i1], accels[i2], accels[i3], accels[i4], accels[i5], accels[i6]};

        short x=0, y=0, vx=0, vy=0;
        for(int s=0;s<6;s++){
            short avgVx = vx + seq[s].ax/2;
            short avgVy = vy + seq[s].ay/2;
            x += avgVx; y += avgVy;
            vx += seq[s].ax; vy += seq[s].ay;
        }

        Result r; r.x=x; r.y=y; r.vx=vx; r.vy=vy;
        for(int s=0;s<6;s++) r.seq[s]=seq[s];
        results[gid]=r;
    }
    )CLC";

    cl::Program program(context, kernelCode);
    program.build({device});
    cl::Kernel kernel(program, "simulate");

    size_t chunkSize = 1 << 20; // 1M sequences per chunk
    size_t totalChunks = (totalSequences + chunkSize - 1) / chunkSize;
    std::atomic<size_t> chunksDone(0);

    // Start progress monitor thread
    std::thread monitor(showProgress, std::ref(chunksDone), totalChunks);

    for(size_t offset = 0; offset < totalSequences; offset += chunkSize) {
        size_t currentChunk = std::min(chunkSize, totalSequences - offset);
        std::vector<Result> chunkResults(currentChunk);
        cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY, sizeof(Result)*currentChunk);

        kernel.setArg(0, accelBuf);
        kernel.setArg(1, (int)N);
        kernel.setArg(2, resultBuf);
        kernel.setArg(3, (int)offset);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(currentChunk));
        queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, sizeof(Result)*currentChunk, chunkResults.data());

        mergeTopX(globalTop, chunkResults, topX);
        chunksDone++;
    }

    monitor.join();

    std::cout << "Top " << topX << " sequences:\n";
    for(auto &r : globalTop)
        std::cout << "Pos=(" << r.x << "," << r.y << "), Vel=(" << r.vx << "," << r.vy << ")\n";
}
