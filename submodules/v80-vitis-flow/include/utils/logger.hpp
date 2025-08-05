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

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <bitset>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace utils {

/**
 * @brief Enumeration of available log levels.
 *
 * These log levels control the verbosity of logging output.
 */
enum class LogLevel {
    INFO,   ///< Informational messages for general reporting
    ERROR,  ///< Error messages indicating failures or issues
    DEBUG   ///< Detailed debug information for troubleshooting
};

/**
 * @brief Static class providing logging facilities.
 *
 * This class implements a flexible logging system with support for
 * different log levels, formatted output, and redirection to files.
 * It supports colored output and custom formatting options.
 */
class Logger {
   public:
    /**
     * @brief Sets the current log level.
     * @param level The log level to set.
     *
     * Only messages at or above the specified level will be logged.
     */
    static void setLogLevel(LogLevel level);

    /**
     * @brief Sets the output destination for log messages.
     * @param filename Path to the file where logs should be written.
     *
     * If this method is called, logs will be written to the specified file
     * instead of the default standard output.
     */
    static void setOutput(const std::string& filename);

    /**
     * @brief Logs a formatted message.
     * @param level The log level for this message.
     * @param function The function name from where this log is called.
     * @param format Format string with placeholders for variables.
     * @param args Variable arguments to include in the formatted string.
     *
     * The format string uses {} for regular value insertion,
     * {x} for hexadecimal, {b} for binary, and {o} for octal formatting.
     *
     * @tparam Args Variadic template parameter for formatting arguments.
     */
    template <typename... Args>
    static void log(LogLevel level, const char* function, const char* format, Args... args) {
        if (level > currentLogLevel_) {
            return;
        }
        std::string color = getColor(level);
        std::string resetColor = (useColours_) ? "\033[0m" : "";
        std::string levelStr = getLevelString(level);
        std::string currentTime = getCurrentTime();
        std::string message = formatString(format, std::forward<Args>(args)...);
        (*output_) << color << "[" << currentTime << "] [" << std::setw(5) << std::left << levelStr
                   << "] " << std::setw(80) << std::left << function << resetColor << ": "
                   << message << std::endl;
    }

   private:
    static std::unique_ptr<std::ofstream> fileStream_;  ///< File stream for log output
    static std::ostream* output_;                       ///< Current output stream
    static bool useColours_;                            ///< Flag to enable/disable colored output
    static LogLevel currentLogLevel_;                   ///< Current log level threshold

    /**
     * @brief Gets the ANSI color code for a log level.
     * @param level The log level.
     * @return String containing the ANSI color code.
     */
    static std::string getColor(LogLevel level);

    /**
     * @brief Gets the string representation of a log level.
     * @param level The log level.
     * @return String containing the log level name.
     */
    static std::string getLevelString(LogLevel level);

    /**
     * @brief Gets the current time as a formatted string.
     * @return String containing the current time.
     */
    static std::string getCurrentTime();

    /**
     * @brief Helper function for string formatting (base case).
     * @param oss Output string stream for the formatted result.
     * @param format Format string remainder.
     */
    static inline void formatStringHelper(std::ostringstream& oss, const char* format) {
        while (*format) {
            if (*format == '{' && *(format + 1) == '}') {
                throw std::runtime_error("Too few arguments provided for format string");
            }
            oss << *format++;
        }
    }

    /**
     * @brief Helper function for string formatting (recursive case).
     * @param oss Output string stream for the formatted result.
     * @param format Format string remainder.
     * @param value Current value to insert.
     * @param args Remaining arguments to format.
     *
     * @tparam T Type of the current value.
     * @tparam Args Types of the remaining arguments.
     */
    template <typename T, typename... Args>
    static void formatStringHelper(std::ostringstream& oss, const char* format, T value,
                                   Args&&... args) {
        while (*format) {
            if (*format == '{' && *(format + 1) == '}') {
                oss << value;
                format += 2;
                formatStringHelper(oss, format, std::forward<Args>(args)...);
                return;
            } else if (*format == '{' && *(format + 1) == 'x' && *(format + 2) == '}') {
                oss << std::hex << std::showbase << value;
                format += 3;
                formatStringHelper(oss, format, std::forward<Args>(args)...);
                return;
            } else if (*format == '{' && *(format + 1) == 'b' && *(format + 2) == '}') {
                oss << "0b" << std::bitset<sizeof(T) * 8>(value);
                format += 3;
                formatStringHelper(oss, format, std::forward<Args>(args)...);
                return;
            } else if (*format == '{' && *(format + 1) == 'o' && *(format + 2) == '}') {
                oss << "0o" << std::oct << std::showbase << value;
                format += 3;
                formatStringHelper(oss, format, std::forward<Args>(args)...);
                return;
            }
            oss << *format++;
        }
        throw std::runtime_error("Too many arguments provided for format string");
    }

    /**
     * @brief Formats a string with the given arguments.
     * @param format Format string with placeholders.
     * @param args Values to insert into the placeholders.
     * @return Formatted string result.
     *
     * @tparam Args Types of the arguments for formatting.
     */
    template <typename... Args>
    static std::string formatString(const char* format, Args&&... args) {
        std::ostringstream oss;
        formatStringHelper(oss, format, std::forward<Args>(args)...);
        return oss.str();
    }
};

}  // namespace utils

#endif  // LOGGER_HPP