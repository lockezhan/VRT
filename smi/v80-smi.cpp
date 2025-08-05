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

#include "arg_parser.hpp"
#include "commands/inspect_command.hpp"
#include "commands/list_command.hpp"
#include "commands/partial_program_command.hpp"
#include "commands/program_command.hpp"
#include "commands/query_command.hpp"
#include "commands/reload_command.hpp"
#include "commands/reset_command.hpp"
#include "commands/resource_command.hpp"
#include "commands/validate_command.hpp"

int main(int argc, char* argv[]) {
    ArgParser parser;
    parser.parse(argc, argv);

    if (parser.isCommand("query")) {
        std::string device = parser.getDevice();
        if (!device.empty()) {
            QueryCommand queryCommand(device);
            queryCommand.execute();
        } else {
            std::cerr << "Error: Device not specified" << std::endl;
            parser.printHelp();
            return EXIT_FAILURE;
        }
    } else if (parser.isCommand("validate")) {
        std::cout << "Validating device..." << std::endl;
        ValidateCommand validateCommand(parser.getDevice());
        validateCommand.execute();
    } else if (parser.isCommand("report_utilization")) {
        ResourceCommand resourceCommand(parser.getDevice());
    } else if (parser.isCommand("list")) {
        ListCommand listCommand(0x10ee, 0x50b4);
        listCommand.execute();
    } else if (parser.isCommand("program")) {
        ProgramCommand programCommand(parser.getDevice(), parser.getImagePath(),
                                      parser.getPartition());
        programCommand.execute();
    } else if (parser.isCommand("partial_program")) {
        PartialProgramCommand partialProgramCommand(parser.getDevice(), parser.getImagePath());
        partialProgramCommand.execute();
    } else if (parser.isCommand("inspect")) {
        InspectCommand inspectCommand(parser.getImagePath());
        inspectCommand.execute();
    } else if (parser.isCommand("reload")) {
        ReloadCommand reloadCommand(parser.getDevice());
        reloadCommand.execute();
    } else if (parser.isCommand("reset")) {
        ResetCommand resetCommand(parser.getDevice());
        resetCommand.execute();
    } else {
        parser.printHelp();
    }

    return EXIT_SUCCESS;
}