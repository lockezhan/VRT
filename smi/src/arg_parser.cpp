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

#include "arg_parser.hpp"

#include <getopt.h>

#include <iostream>
#include <regex>

ArgParser::ArgParser() {
    addCommand("query", [this]() { currentCommand = "query"; });
    addCommand("validate", [this]() { currentCommand = "validate"; });
    addCommand("report_utilization", [this]() { currentCommand = "report_utilization"; });
    addCommand("list", [this]() { currentCommand = "list"; });
    addCommand("program", [this]() { currentCommand = "program"; });
    addCommand("partial_program", [this]() { currentCommand = "partial_program"; });
    addCommand("inspect", [this]() { currentCommand = "inspect"; });
    addCommand("reload", [this]() { currentCommand = "reload"; });
    addCommand("reset", [this]() { currentCommand = "reset"; });
}

void ArgParser::parse(int argc, char* argv[]) {
    static struct option long_options[] = {{"device", required_argument, 0, 'd'},
                                           {"image", required_argument, 0, 'i'},
                                           {"partition", required_argument, 0, 'p'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    bool isProgram = false;
    bool isPartialProgram = false;
    bool isInspect = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "program") {
            isProgram = true;
            break;
        }
    }
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "partial_program") {
            isPartialProgram = true;
            break;
        }
    }

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "inspect") {
            isInspect = true;
            break;
        }
    }
    while ((opt = getopt_long(argc, argv, "d:i:p:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd':
                device = convertBdf(optarg);
                break;
            case 'i':
                image = optarg;
                break;
            case 'p':
                partition = std::stoi(optarg);
                break;
            case 'h':
                printHelp();
                exit(EXIT_SUCCESS);
            case '?':
                printHelp();
                exit(EXIT_FAILURE);
            default:
                break;
        }
    }

    if (optind < argc) {
        std::string command = argv[optind];
        auto it = commands.find(command);
        if (it != commands.end()) {
            it->second();
        } else {
            std::cerr << "Error: Unknown command " << command << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
    } else {
        std::cerr << "Error: No command specified" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
    }
    if (isProgram) {
        if (device.empty() || image.empty() || partition == -1) {
            std::cerr << "Error: Missing required options for 'program' command." << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
        if (partition > 1) {
            std::cerr << "Partition must be 0 or 1" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!(endsWith(image, ".vrtbin"))) {
            std::cerr << "Image must be a .vrtbin file" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!std::filesystem::exists(image)) {
            std::cerr << "Image file does not exist" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (isPartialProgram) {
        if (device.empty() || image.empty() || partition == -1) {
            std::cerr << "Error: Missing required options for 'partial_program' command."
                      << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }

        if (!((endsWith(image, ".vrtbin")) || (endsWith(image, ".pdi")))) {
            std::cerr << "Image must be a .vrtbin or pdi file" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!std::filesystem::exists(image)) {
            std::cerr << "Image file does not exist" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (isInspect) {
        if (image.empty()) {
            std::cerr << "Error: Missing required options for 'inspect' command." << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }

        if (!(endsWith(image, ".vrtbin"))) {
            std::cerr << "Image must be a .vrtbin file" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!std::filesystem::exists(image)) {
            std::cerr << "Image file does not exist" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

std::string ArgParser::getDevice() const { return device; }

bool ArgParser::isCommand(const std::string& command) const { return currentCommand == command; }

void ArgParser::printHelp() const {
    std::cout
        << "Usage: v80-smi <command> [options]\n"
        << "Commands:\n"
        << "  query                Query the device\n"
        << "  validate             Validate the device\n"
        << "  report_utilization   Report device utilization for the current programmed shell\n"
        << "  list                 List V80s installed\n"
        << "  program              Program the device's flash memory\n"
        << "  partial_program      Program the device with a segmented PDI image\n"
        << "  inspect              Inspect a vrtbin before programming\n"
        << "  reload               Reloads the PCIe handler for device\n"
        << "  reset                Resets the device to a clean state\n"
        << "Options:\n"
        << "  -d, --device <device>  Specify the device (e.g., 21:00.0)\n"
        << "  -i, --image <image>    Specify the image file to program. Only relevant for "
           "program/partial_program commands\n"
        << "  -p, --partition <num>  Specify the partition to program. Only relevant for program "
           "command\n"
        << "  -h, --help             Show this help message\n";
}

void ArgParser::addCommand(const std::string& command, const std::function<void()>& handler) {
    commands[command] = handler;
}

std::string ArgParser::convertBdf(const std::string& bdf) const {
    std::regex pattern("^0000:(.*)");
    std::smatch match;
    if (std::regex_match(bdf, match, pattern)) {
        return match[1].str();
    }
    return strip(bdf);
}

std::string ArgParser::strip(const std::string& bdf) const {
    size_t colonPos = bdf.find(':');
    if (colonPos != std::string::npos) {
        return bdf.substr(0, colonPos);
    }
    return bdf;
}

std::string ArgParser::getImagePath() const { return image; }

uint8_t ArgParser::getPartition() const { return partition; }

bool ArgParser::endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
