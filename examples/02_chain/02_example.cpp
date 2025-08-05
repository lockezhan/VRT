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
#include <cstdint>
#include <chrono>
#include <utils/logger.hpp>
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
        uint32_t size = 1024;
        vrt::utils::Logger::setLogLevel(vrt::utils::LogLevel::DEBUG);
        vrt::Device device(bdf, vrtbinFile, false);  // program=false, 不执行烧录

        vrt::Kernel dma_in(device, "dma_in_0");
        vrt::Kernel dma_out(device, "dma_out_0");
        vrt::Buffer<uint64_t> buffer_in(device, size, vrt::MemoryRangeType::HBM);
        vrt::Buffer<uint64_t> buffer_out(device, size, vrt::MemoryRangeType::HBM);
        for(uint32_t i = 0; i < size; i++) {
            buffer_in[i] = static_cast<uint64_t>(i);
        }
        buffer_in.sync(vrt::SyncType::HOST_TO_DEVICE);
        
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        dma_in.start(buffer_in.getPhysAddr(), size);
        dma_out.start(size, buffer_out.getPhysAddr());
        dma_in.wait();
        dma_out.wait();
        
        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // 计算执行时间
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "DMA chain execution time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "DMA chain execution time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
        
        buffer_out.sync(vrt::SyncType::DEVICE_TO_HOST);
        for(uint32_t i = 0; i < size; i++) {
            if(buffer_in[i] != buffer_out[i]) {
                vrt::utils::Logger::log(vrt::utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Test failed");
                device.cleanup();
                return 0;
            }
        }
        vrt::utils::Logger::log(vrt::utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Test passed");
        device.cleanup();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
