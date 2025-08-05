/**
 * The MIT License (MIT)
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <cstring>
#include <vector>
#include <chrono> 

#include <api/device.hpp>
#include <api/buffer.hpp>
#include <api/kernel.hpp>

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <BDF> <vrtbin file>" << std::endl;
            return 1;
        }
        std::string bdf = argv[1];
        std::string vrtbinFile = argv[2];
        vrt::utils::Logger::setLogLevel(vrt::utils::LogLevel::DEBUG);
        vrt::Device device("21:00.0", "05_perf_emu.vrtbin");
        uint32_t size = 1024;
        device.setFrequency(400000000);

        std::vector<vrt::Kernel> kernels;
        kernels.reserve(15);
        for(int i = 0; i < 15; i++) {
            kernels.push_back(std::move(vrt::Kernel(device, "perf_" + std::to_string(i))));
        }

        std::vector<vrt::Buffer<int>> buffers;
        buffers.reserve(15);
        for(int i = 0; i < 15; i++) {
            vrt::Buffer<int> buffer(device, size, vrt::MemoryRangeType::HBM);
            buffers.emplace_back(std::move(buffer));
        }

        for (int i = 0; i < 15; i++) {
            uint64_t addr = buffers[i].getPhysAddr();
            kernels[i].start(size * sizeof(uint32_t) / 64, buffers[i].getPhysAddr());
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        // The line you want to measure
        kernels[14].wait();
        
        // Add timing code after wait call
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Print the time taken
        std::cout << "Time taken for kernels[14].wait(): " << duration.count() << " milliseconds" << std::endl;

        device.cleanup();
    } catch (const std::exception& e) {
        // std::cerr << e.what() << std::endl;
        // device.cleanup();
    }

    // vrt::Device device("21:00.0", "05_perf_emu.vrtbin");
    // try {
    //     uint32_t size = 1000;
    //     vrt::Kernel perf_0(device, "perf_0");
    //     vrt::Buffer<int> buffer(device, size, vrt::MemoryRangeType::HBM);
    //     for(int j = 0; j < size; j++) {
    //         buffer[j] = j;
    //     }
    //     buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
    //     perf_0.call(size, buffer.getPhysAddr());
    //     buffer.sync(vrt::SyncType::DEVICE_TO_HOST);
    //     for(int j = 0; j < size; j++) {
    //         std::cout << buffer[j] << " ";
    //     }
    //     device.cleanup();
    // } catch (const std::exception& e) {
    //     std::cerr << e.what() << std::endl;
    //     device.cleanup();
    // }
    // return 0;
}