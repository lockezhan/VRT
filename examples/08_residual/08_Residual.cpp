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

        device.setFrequency(319454797);  // 提高到400MHz

        // 分配所有输入buffer - 分散到不同HBM端口以提高带宽
        vrt::Buffer<double> x(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 0);
        vrt::Buffer<double> cost(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 1);
        vrt::Buffer<double> ax(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 2);
        vrt::Buffer<double> rhs(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 3);
        vrt::Buffer<double> rowScale(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 4);
        vrt::Buffer<double> aty(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 5);
        vrt::Buffer<double> y(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 6);
        vrt::Buffer<double> hasLower(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 7);
        vrt::Buffer<double> dLowerFiltered(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 8);
        vrt::Buffer<double> hasUpper(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 9);
        vrt::Buffer<double> dUpperFiltered(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 10);
        vrt::Buffer<double> colScale(device, MAX_SIZE, vrt::MemoryRangeType::HBM, 11);
        vrt::Buffer<double> result(device, 4, vrt::MemoryRangeType::HBM, 12); // 输出为4个double

        // 随机初始化输入
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (uint32_t i = 0; i < MAX_SIZE; i++) {
            x[i] = dis(gen);
            cost[i] = dis(gen);
            ax[i] = dis(gen);
            rhs[i] = dis(gen);
            rowScale[i] = dis(gen);
            aty[i] = dis(gen);
            y[i] = dis(gen);
            hasLower[i] = dis(gen);
            dLowerFiltered[i] = dis(gen);
            hasUpper[i] = dis(gen);
            dUpperFiltered[i] = dis(gen);
            colScale[i] = dis(gen);
        }

        // 同步到设备
        x.sync(vrt::SyncType::HOST_TO_DEVICE);
        cost.sync(vrt::SyncType::HOST_TO_DEVICE);
        ax.sync(vrt::SyncType::HOST_TO_DEVICE);
        rhs.sync(vrt::SyncType::HOST_TO_DEVICE);
        rowScale.sync(vrt::SyncType::HOST_TO_DEVICE);
        aty.sync(vrt::SyncType::HOST_TO_DEVICE);
        y.sync(vrt::SyncType::HOST_TO_DEVICE);
        hasLower.sync(vrt::SyncType::HOST_TO_DEVICE);
        dLowerFiltered.sync(vrt::SyncType::HOST_TO_DEVICE);
        hasUpper.sync(vrt::SyncType::HOST_TO_DEVICE);
        dUpperFiltered.sync(vrt::SyncType::HOST_TO_DEVICE);
        colScale.sync(vrt::SyncType::HOST_TO_DEVICE);

        std::cout << "All buffers synced to device." << std::endl;

        vrt::Kernel residuals(device, "Residuals_kernel_0");

        // 启动kernel
        auto start = std::chrono::high_resolution_clock::now();
        residuals.start(
            x.getPhysAddr(), cost.getPhysAddr(), ax.getPhysAddr(), rhs.getPhysAddr(), rowScale.getPhysAddr(),
            aty.getPhysAddr(), y.getPhysAddr(),
            hasLower.getPhysAddr(), dLowerFiltered.getPhysAddr(),
            hasUpper.getPhysAddr(), dUpperFiltered.getPhysAddr(), colScale.getPhysAddr(),
            result.getPhysAddr(),
            nCols, nRows, nEqs
        );
        residuals.wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Kernel execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Total kernel execution time: " << total_time / 1000.0 << " ms" << std::endl;


        // 取回结果
        result.sync(vrt::SyncType::DEVICE_TO_HOST);
        std::cout << "Result[0]: " << result[0] << std::endl;
        std::cout << "Result[1]: " << result[1] << std::endl;
        std::cout << "Result[2]: " << result[2] << std::endl;
        std::cout << "Result[3]: " << result[3] << std::endl;

        device.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}