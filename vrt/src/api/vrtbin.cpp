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

#include "api/vrtbin.hpp"

namespace vrt {
Vrtbin::Vrtbin(std::string vrtbinPath, const std::string& bdf) {
    this->vrtbinPath = vrtbinPath;
    if (!std::filesystem::exists(vrtbinPath)) {
        throw std::runtime_error(vrtbinPath + " does not exist");
    }
    char* ami_home_cstr = getenv("AMI_HOME");
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "AMI_HOME: {}", ami_home_cstr);
    if (ami_home_cstr == nullptr) {
        throw std::runtime_error("AMI_HOME environment variable not set");
    }
    std::string ami_home(getenv("AMI_HOME"));
    if (!ami_home.empty() && ami_home.back() != '/') {
        ami_home += '/';
    }
    std::string cmd = "mkdir -p " + ami_home + bdf;
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "Running command: {}", cmd);
    system(cmd.c_str());
    this->systemMapPath = ami_home + bdf + "/system_map.xml";
    extract();
    std::string tempSystemMapPath = tempExtractPath + "/system_map.xml";
    XMLParser parser(tempSystemMapPath);
    parser.parseXML();
    this->platform = parser.getPlatform();
    if (this->platform == Platform::HARDWARE) {
        this->versionPath = ami_home + bdf + "/version.json";
        this->pdiPath = tempExtractPath + "/design.pdi";
        copy(tempExtractPath + "/system_map.xml", systemMapPath);
        copy(tempExtractPath + "/version.json", versionPath);
        copy(tempExtractPath + "/report_utilization.xml",
             ami_home + bdf + "/report_utilization.xml");
        extractUUID();
    } else if (this->platform == Platform::EMULATION) {
        copy(tempExtractPath + "/system_map.xml", systemMapPath);
        emulationExecPath = tempExtractPath + "/vpp_emu";

    } else {
        copy(tempExtractPath + "/system_map.xml", systemMapPath);
        simulationExecPath = tempExtractPath + "/vpp_sim";
    }
}

void Vrtbin::extract() {
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "Extracting vrtbin: {}",
                       vrtbinPath);
    std::string command = "tar -xvf " + vrtbinPath + " -C " + tempExtractPath + " 2>&1";
    std::array<char, 128> buffer;
    std::string result;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
}

void Vrtbin::copy(const std::string& source, const std::string& destination) {
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "Copying file {} to {}", source,
                       destination);
    std::ifstream src(source, std::ios::binary);
    if (!src) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Error opening source file: {}", source);
        throw std::runtime_error("Error opening source file");
    }

    std::ofstream dest(destination, std::ios::binary);
    if (!dest) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Error opening destination file: {}", destination);
        throw std::runtime_error("Error opening destination file");
    }

    dest << src.rdbuf();

    if (!src) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Error reading from source file: {}", source);
        throw std::runtime_error("Error reading from source file");
    }

    if (!dest) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Error writing to destination file: {}", destination);
        throw std::runtime_error("Error writing to destination file");
    }
}

std::string Vrtbin::getSystemMapPath() { return systemMapPath; }
std::string Vrtbin::getPdiPath() { return pdiPath; }

std::string Vrtbin::getUUID() { return uuid; }

void Vrtbin::extractUUID() {
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                       "Extracting UUID from version.json");
    std::ifstream jsonFile(tempExtractPath + "/version.json");
    if (!jsonFile.is_open()) {
        uuid = "";
    }
    std::string line;
    while (std::getline(jsonFile, line)) {
        std::size_t pos = line.find("\"logic_uuid\":");
        if (pos != std::string::npos) {
            std::size_t start = line.find("\"", pos + 13) + 1;
            std::size_t end = line.find("\"", start);
            uuid = line.substr(start, end - start);
            break;
        }
    }
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "UUID is: {}", uuid);
    jsonFile.close();
}

std::string Vrtbin::getEmulationExec() { return emulationExecPath; }

std::string Vrtbin::getSimulationExec() { return simulationExecPath; }
}  // namespace vrt