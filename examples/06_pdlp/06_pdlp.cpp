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
#include <cstring> // for std::memcpy
#include <random>
#include <chrono>
#include <cmath>   // for std::abs
#include <algorithm> // for std::min

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
        uint32_t size = 1024 * 1024;  // 恢复到大数据量测试
        vrt::utils::Logger::setLogLevel(vrt::utils::LogLevel::INFO);
        std::cout << "VRT Version: " << vrt::getVersion() << std::endl;
        vrt::Device device(bdf, vrtbinFile, false);  // program=false, 不执行烧录

        vrt::Kernel axpy(device, "axpy_0");
        vrt::Kernel stream_buffer(device, "stream_buffer_0");
        vrt::Kernel projlub(device, "projlub_0");
        
        // Create three buffers for axpy inputs
        vrt::Buffer<double> x_in_buffer(device, size, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> aty_buffer(device, size, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> cost_buffer(device, size, vrt::MemoryRangeType::HBM);
        
        // Create buffers for projlub
        vrt::Buffer<double> lb_buffer(device, size, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> ub_buffer(device, size, vrt::MemoryRangeType::HBM);
        vrt::Buffer<double> xUpdate_buffer(device, size, vrt::MemoryRangeType::HBM);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        double alpha = 0.5; // Set alpha value for axpy
        std::cout << "Generating data...\n";
        for(uint32_t i = 0; i < size; i++) {
            x_in_buffer[i] = static_cast<double>(dis(gen));
            aty_buffer[i] = static_cast<double>(dis(gen));
            cost_buffer[i] = static_cast<double>(dis(gen));
            lb_buffer[i] = static_cast<double>(dis(gen)) - 1.0; // Lower bound
            ub_buffer[i] = static_cast<double>(dis(gen)) + 1.0; // Upper bound
        }

        // Sync all input buffers to device
        std::cout << "Syncing buffers to device...\n";  

        x_in_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
        aty_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
        cost_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
        lb_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
        ub_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);

        std::cout << "All buffer syncs completed!\n";
        
        // Start kernels - stream_buffer should start first as it's always ready
        // Convert double alpha to uint64_t for kernel parameter passing
        uint64_t alpha_bits;
        std::memcpy(&alpha_bits, &alpha, sizeof(double));
        
        axpy.start(x_in_buffer.getPhysAddr(), aty_buffer.getPhysAddr(), cost_buffer.getPhysAddr(), alpha_bits, size);
        std::cout << "axpy.start() completed.\n";
        projlub.start(lb_buffer.getPhysAddr(), ub_buffer.getPhysAddr(), xUpdate_buffer.getPhysAddr(), size);
        std::cout << "projlub.start() completed.\n";
        

        // Wait for both kernels to complete
        std::cout << "Waiting for both kernels to complete...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Waiting for axpy kernel...\n";
        axpy.wait();
        auto axpy_end = std::chrono::high_resolution_clock::now();
        auto axpy_duration = std::chrono::duration_cast<std::chrono::microseconds>(axpy_end - start_time).count();
        std::cout << "axpy kernel completed in " << axpy_duration << " us.\n";
        
        std::cout << "Waiting for projlub kernel...\n";
        projlub.wait();
        auto projlub_end = std::chrono::high_resolution_clock::now();
        auto projlub_duration = std::chrono::duration_cast<std::chrono::microseconds>(projlub_end - start_time).count();
        std::cout << "projlub kernel completed in " << projlub_duration << " us.\n";

        auto total_duration = std::max(axpy_duration, projlub_duration);
        std::cout << "Total kernel execution time: " << total_duration << " us" << std::endl;

        // Sync result buffer back to host
        xUpdate_buffer.sync(vrt::SyncType::DEVICE_TO_HOST);
        
        // Calculate golden model and verify results
        std::cout << "Calculating golden model and verifying results..." << std::endl;
        bool allPassed = true;
        int passCount = 0;
        int failCount = 0;
        
        for(uint32_t i = 0; i < size; i++) {
            // Golden model calculation: Xin - alpha*(cost-aty)
            double axpy_result = x_in_buffer[i] - alpha * (cost_buffer[i] - aty_buffer[i]);
            
            // Apply projection bounds: lb <= result <= ub
            double golden_result = axpy_result;
            if (golden_result < lb_buffer[i]) {
                golden_result = lb_buffer[i];  // Clamp to lower bound
            } else if (golden_result > ub_buffer[i]) {
                golden_result = ub_buffer[i];  // Clamp to upper bound
            }
            
            // Compare with FPGA result
            double fpga_result = xUpdate_buffer[i];
            double error = std::abs(golden_result - fpga_result);
            
            if (error > 1e-6) {  // Tolerance for double precision
                if (failCount < 10) {  // Only print first 10 failures
                    std::cout << "FAIL[" << i << "]: Golden=" << golden_result 
                              << ", FPGA=" << fpga_result << ", Error=" << error << std::endl;
                }
                allPassed = false;
                failCount++;
            } else {
                passCount++;
            }
        }
        
        std::cout << "\n=== Verification Results ===" << std::endl;
        std::cout << "Total elements: " << size << std::endl;
        std::cout << "Passed: " << passCount << std::endl;
        std::cout << "Failed: " << failCount << std::endl;
        std::cout << "Success rate: " << (100.0 * passCount / size) << "%" << std::endl;
        
        if (allPassed) {
            std::cout << "✓ All tests PASSED!" << std::endl;
        } else {
            std::cout << "✗ Some tests FAILED!" << std::endl;
        }
        
        std::cout << "\nFirst few results comparison:" << std::endl;
        for(int i = 0; i < std::min(10, (int)size); i++) {
            double axpy_result = x_in_buffer[i] - alpha * (cost_buffer[i] - aty_buffer[i]);
            double golden_result = axpy_result;
            if (golden_result < lb_buffer[i]) golden_result = lb_buffer[i];
            else if (golden_result > ub_buffer[i]) golden_result = ub_buffer[i];
            
            std::cout << "[" << i << "] FPGA=" << xUpdate_buffer[i] 
                      << ", Golden=" << golden_result 
                      << ", Error=" << std::abs(golden_result - xUpdate_buffer[i]) << std::endl;
        }
        
        device.cleanup();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } 
    return 0;
}