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

#pragma once

/**
 * @file defines.hpp
 * @brief Constants for XML parsing in the Vitis Flow.
 *
 * This header file defines constants used for parsing XML files
 * that contain hardware kernel definitions. These constants specify
 * XML node names, attribute names, and other parsing parameters
 * used throughout the application.
 */

/**
 * @brief XML node name for version information.
 */
#define XML_NODE_VERSION "Version"

/**
 * @brief XML node name for time unit specification.
 */
#define XML_NODE_UNIT "unit"

/**
 * @brief XML node name for FPGA product family.
 */
#define XML_NODE_PRODUCT_FAMILY "ProductFamily"

/**
 * @brief XML node name for FPGA part number.
 */
#define XML_NODE_PART "Part"

/**
 * @brief XML node name for top-level model name.
 */
#define XML_NODE_TOP_MODEL_NAME "TopModelName"

/**
 * @brief XML node name for target clock period.
 */
#define XML_NODE_TARGET_CLK "TargetClockPeriod"

/**
 * @brief XML node name for clock uncertainty.
 */
#define XML_NODE_CLK_UNCERTAINTY "ClockUncertainty"

/**
 * @brief XML node name for estimated clock period.
 */
#define XML_NODE_ESTIMATED_CLK "EstimatedClockPeriod"

/**
 * @brief XML node name for 18K BRAM resource.
 */
#define XML_NODE_BRAM_18K "BRAM_18K"

/**
 * @brief XML node name for flip-flop (FF) resource.
 */
#define XML_NODE_FF "FF"

/**
 * @brief XML node name for look-up table (LUT) resource.
 */
#define XML_NODE_LUT "LUT"

/**
 * @brief XML node name for ultra RAM (URAM) resource.
 */
#define XML_NODE_URAM "URAM"

/**
 * @brief XML node name for DSP resource.
 */
#define XML_NODE_DSP "DSP"

/**
 * @brief XML node name for interface definitions.
 */
#define XML_NODE_INTERFACE "Interface"

/**
 * @brief XML node name for register definitions.
 */
#define XML_NODE_REGISTER "register"

/**
 * @brief XML attribute name for interface name.
 */
#define XML_ATTR_INTF_NAME "InterfaceName"

/**
 * @brief XML attribute name for type specification.
 */
#define XML_ATTR_TYPE "type"

/**
 * @brief XML attribute name for bus type.
 */
#define XML_ATTR_BUS_TYPE "busTypeName"

/**
 * @brief XML attribute name for interface mode.
 */
#define XML_ATTR_MODE "mode"

/**
 * @brief XML attribute name for data width.
 */
#define XML_ATTR_DATA_WIDTH "dataWidth"

/**
 * @brief XML attribute name for address width.
 */
#define XML_ATTR_ADDR_WIDTH "addrWidth"