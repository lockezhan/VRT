#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <utils/logger.hpp>
#include <api/device.hpp>
#include <api/buffer.hpp>
#include <api/kernel.hpp>

const uint32_t MAX_SIZE = 1e6; 

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <BDF> <vrtbin file>" << std::endl;
            return 1;
        }
        std::string bdf = argv[1];
        std::string vrtbinFile = argv[2];

        // 假设总元素个数为N，vec_double宽度为8
        uint32_t nCols = MAX_SIZE;
        uint32_t nRows = MAX_SIZE;
        uint32_t nEqs  = 0;
        uint32_t vec_size = 8;
        uint32_t loop_num = (nCols + vec_size - 1) / vec_size;

        vrt::utils::Logger::setLogLevel(vrt::utils::LogLevel::INFO);
        vrt::Device device(bdf, vrtbinFile, false);

        device.setFrequency(300000000);  // 300MHz - 与配置文件一致

        // 分配输入 buffer - 对应 read_buf 的输入接口 (HBM 0-6)
        vrt::Buffer<double> cost(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 0);
        vrt::Buffer<double> aty(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 1);
        vrt::Buffer<double> hasLower(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 2);
        vrt::Buffer<double> dLowerFiltered(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 3);
        vrt::Buffer<double> hasUpper(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 4);
        vrt::Buffer<double> dUpperFiltered(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 5);
        vrt::Buffer<double> colScale(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 6);
        
        // 分配输出 buffer - 两个 compute_lambda 的结果 (HBM 7-8)
        vrt::Buffer<double> result0(device, 4, vrt::MemoryRangeType::HBM, 7); // 第一个 compute_lambda 结果
        vrt::Buffer<double> result1(device, 4, vrt::MemoryRangeType::HBM, 8); // 第二个 compute_lambda 结果

        std::cout << "Allocated buffers:" << std::endl;
        std::cout << "  Input data size: " << MAX_SIZE << " elements per array" << std::endl;
        std::cout << "  Vector size: " << vec_size << std::endl;
        std::cout << "  Loop iterations: " << loop_num << std::endl;

        // 随机初始化输入数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        std::cout << "Initializing input data..." << std::endl;
        for (uint32_t i = 0; i < MAX_SIZE; i++) {
            cost[i] = dis(gen);
            aty[i] = dis(gen);
            hasLower[i] = dis(gen) > 0.5 ? 1.0 : 0.0;  // 二进制标志
            dLowerFiltered[i] = dis(gen);
            hasUpper[i] = dis(gen) > 0.5 ? 1.0 : 0.0;  // 二进制标志
            dUpperFiltered[i] = dis(gen);
            colScale[i] = dis(gen) + 0.1;  // 避免除零，最小值0.1
        }

        // 同步输入数据到设备
        std::cout << "Syncing input data to device..." << std::endl;
        cost.sync(vrt::SyncType::HOST_TO_DEVICE);
        aty.sync(vrt::SyncType::HOST_TO_DEVICE);
        hasLower.sync(vrt::SyncType::HOST_TO_DEVICE);
        dLowerFiltered.sync(vrt::SyncType::HOST_TO_DEVICE);
        hasUpper.sync(vrt::SyncType::HOST_TO_DEVICE);
        dUpperFiltered.sync(vrt::SyncType::HOST_TO_DEVICE);
        colScale.sync(vrt::SyncType::HOST_TO_DEVICE);

        std::cout << "All input buffers synced to device." << std::endl;

        // 创建内核实例 - 对应我们配置的三个内核
        vrt::Kernel read_buf_kernel(device, "read_buf_0");
        vrt::Kernel compute_lambda_0(device, "compute_lambda_0");
        vrt::Kernel compute_lambda_1(device, "compute_lambda_1");

        std::cout << "Starting dual compute_lambda execution..." << std::endl;
        std::cout << "Architecture: read_buf -> [Stream] -> 2x compute_lambda" << std::endl;
        
        // 启动 read_buf 内核 (数据分发器)
        std::cout << "Starting read_buf kernel..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        read_buf_kernel.start(
            cost.getPhysAddr(),           // cost -> HBM[0]
            aty.getPhysAddr(),            // aty -> HBM[1]
            hasLower.getPhysAddr(),       // hasLower -> HBM[2]
            dLowerFiltered.getPhysAddr(), // dLowerFiltered -> HBM[3]
            hasUpper.getPhysAddr(),       // hasUpper -> HBM[4]
            dUpperFiltered.getPhysAddr(), // dUpperFiltered -> HBM[5]
            colScale.getPhysAddr(),       // colScale -> HBM[6]
            nCols, nRows, nEqs
        );

        // 启动两个 compute_lambda 内核 (并行执行)
        std::cout << "Starting compute_lambda_0 kernel..." << std::endl;
        compute_lambda_0.start(
            result0.getPhysAddr(),        // result0 -> HBM[7]
            nCols
        );

        std::cout << "Starting compute_lambda_1 kernel..." << std::endl;
        compute_lambda_1.start(
            result1.getPhysAddr(),        // result1 -> HBM[8]
            nCols
        );

        // 等待所有内核完成
        std::cout << "Waiting for read_buf completion..." << std::endl;
        read_buf_kernel.wait();
        
        std::cout << "Waiting for compute_lambda_0 completion..." << std::endl;
        compute_lambda_0.wait();
        
        std::cout << "Waiting for compute_lambda_1 completion..." << std::endl;
        compute_lambda_1.wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "=== Execution Completed ===" << std::endl;
        std::cout << "Total execution time: " << total_time << " us" << std::endl;
        std::cout << "Total execution time: " << total_time / 1000.0 << " ms" << std::endl;


        // 同步结果数据回Host
        std::cout << "Syncing results from device..." << std::endl;
        result0.sync(vrt::SyncType::DEVICE_TO_HOST);
        result1.sync(vrt::SyncType::DEVICE_TO_HOST);

        // 显示两个 compute_lambda IP 的结果
        std::cout << "\n=== Results from compute_lambda_0 (IP0) ===" << std::endl;
        std::cout << "Result0[0] (dPrimalObj): " << result0[0] << std::endl;
        std::cout << "Result0[1] (dPrimalFeasibility): " << result0[1] << std::endl;
        std::cout << "Result0[2] (dPrimalObj_computed): " << result0[2] << std::endl;
        std::cout << "Result0[3] (dDualFeasibility): " << result0[3] << std::endl;

        std::cout << "\n=== Results from compute_lambda_1 (IP1) ===" << std::endl;
        std::cout << "Result1[0] (dPrimalObj): " << result1[0] << std::endl;
        std::cout << "Result1[1] (dPrimalFeasibility): " << result1[1] << std::endl;
        std::cout << "Result1[2] (dPrimalObj_computed): " << result1[2] << std::endl;
        std::cout << "Result1[3] (dDualFeasibility): " << result1[3] << std::endl;

        // 分析数据分发效果
        std::cout << "\n=== Data Distribution Analysis ===" << std::endl;
        int expected_ip0_elements = (loop_num + 1) / 2;  // 偶数索引元素数
        int expected_ip1_elements = loop_num / 2;        // 奇数索引元素数
        std::cout << "Expected data distribution (alternating mode):" << std::endl;
        std::cout << "  IP0 (even indices): " << expected_ip0_elements << " vector blocks" << std::endl;
        std::cout << "  IP1 (odd indices):  " << expected_ip1_elements << " vector blocks" << std::endl;
        std::cout << "  Total vector blocks: " << loop_num << std::endl;
        std::cout << "  Load balance ratio: " << (double)expected_ip0_elements / expected_ip1_elements << std::endl;

        // 性能分析
        std::cout << "\n=== Performance Analysis ===" << std::endl;
        double throughput_gbps = (MAX_SIZE * 7 * sizeof(double)) / (total_time / 1e6) / 1e9;  // 7个输入数组
        std::cout << "Data throughput: " << throughput_gbps << " GB/s" << std::endl;
        std::cout << "Elements per second: " << (MAX_SIZE * 7) / (total_time / 1e6) / 1e6 << " M elements/s" << std::endl;
        
        if (total_time > 0) {
            double theoretical_peak = 300e6 * 8 * sizeof(double) * 7 / 1e9;  // 300MHz * VEC_SIZE * 7arrays
            std::cout << "Theoretical peak: " << theoretical_peak << " GB/s" << std::endl;
            std::cout << "Efficiency: " << (throughput_gbps / theoretical_peak) * 100 << "%" << std::endl;
        }

        device.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}