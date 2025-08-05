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

#ifndef ARG_PARSER_HPP
#define ARG_PARSER_HPP

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>

/**
 * @brief Class for parsing command-line arguments.
 *
 * The ArgParser class handles the parsing of command-line arguments for the application.
 */
class ArgParser {
   public:
    /**
     * @brief Constructor for ArgParser.
     *
     * Initializes the ArgParser and registers the available commands.
     */
    ArgParser();

    /**
     * @brief Parses the command-line arguments.
     *
     * @param argc The number of arguments.
     * @param argv The array of argument strings.
     */
    void parse(int argc, char* argv[]);

    /**
     * @brief Gets the device BDF (Bus:Device.Function) identifier.
     *
     * @return The device BDF as a string.
     */
    std::string getDevice() const;

    /**
     * @brief Gets the path to the image file.
     *
     * @return The image file path as a string.
     */
    std::string getImagePath() const;

    /**
     * @brief Gets the partition number.
     *
     * @return The partition number.
     */
    uint8_t getPartition() const;

    /**
     * @brief Checks if a specific command was specified.
     *
     * @param command The command to check for.
     * @return True if the specified command was given, false otherwise.
     */
    bool isCommand(const std::string& command) const;

    /**
     * @brief Prints the help message.
     *
     * Displays usage information, available commands, and options.
     */
    void printHelp() const;

    /**
     * @brief Checks if a string ends with a specific suffix.
     *
     * @param str The string to check.
     * @param suffix The suffix to look for.
     * @return True if the string ends with the suffix, false otherwise.
     */
    static bool endsWith(const std::string& str, const std::string& suffix);

   private:
    std::unordered_map<std::string, std::function<void()>>
        commands;                ///< Map of command names to handler functions.
    std::string device;          ///< The device BDF identifier.
    std::string image;           ///< The path to the image file.
    uint8_t partition = -1;      ///< The partition number.
    std::string currentCommand;  ///< The currently active command.

    /**
     * @brief Registers a command with its handler function.
     *
     * @param command The command name.
     * @param handler The function to execute when the command is invoked.
     */
    void addCommand(const std::string& command, const std::function<void()>& handler);

    /**
     * @brief Converts a BDF string to a standardized format.
     *
     * @param bdf The BDF string to convert.
     * @return The standardized BDF string.
     */
    std::string convertBdf(const std::string& bdf) const;

    /**
     * @brief Strips whitespace from a string.
     *
     * @param bdf The string to strip.
     * @return The stripped string.
     */
    std::string strip(const std::string& bdf) const;
};

#endif  // ARG_PARSER_HPP