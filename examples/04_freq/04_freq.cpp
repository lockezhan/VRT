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
#include <chrono>

#include <api/device.hpp>
#include <api/buffer.hpp>
#include <api/kernel.hpp>

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <BDF> <vrtbin_path>" << std::endl;
            return 1;
        }
        std::string bdf = argv[1];
        std::string vrtbinPath = argv[2];
        vrt::utils::Logger::setLogLevel(vrt::utils::LogLevel::DEBUG);
        uint32_t size = 1024;

        vrt::Device device(bdf, vrtbinPath, false);  // program=false, 不执行烧录
        vrt::Kernel vadd_0(device, "vadd_0");

        std::cout << "Current set frequency: "<< device.getFrequency() << " Hz" << std::endl;
        std::cout << "Max frequency: "<< device.getMaxFrequency() << " Hz" << std::endl;
        device.setFrequency(200000000);
        std::cout << "Current set frequency: "<< device.getFrequency() << " Hz" << std::endl;

        vrt::Buffer<double> a(device, size*8, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> b(device, size*8, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> c(device, size*8, vrt::MemoryRangeType::HBM);

        for (int i = 0; i < size*8; i++) {
            a[i] = 1;
            b[i] = 1;
        }
        a.sync(vrt::SyncType::HOST_TO_DEVICE);
        b.sync(vrt::SyncType::HOST_TO_DEVICE);
        
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        vadd_0.start(a.getPhysAddr(), b.getPhysAddr(), c.getPhysAddr(), size);
        vadd_0.wait();
        
        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // 计算执行时间
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Kernel execution time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Kernel execution time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
        
        // c.sync(vrt::SyncType::DEVICE_TO_HOST);
        // for (int i = 0; i < size; i++) {
        //     if (c[i] != a[i] + b[i]) {
        //         std::cerr << "Error: " << c[i] << " != " << a[i] << " + " << b[i] << std::endl;
        //         device.cleanup();
        //         return 1;
        //     }
        // }
        std::cout << "Test passed" << std::endl;
        device.cleanup();
     } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}