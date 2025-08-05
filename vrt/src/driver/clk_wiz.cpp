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

#include "driver/clk_wiz.hpp"

namespace vrt {

ClkWiz::ClkWiz(ami_device* device, const std::string& name, uint64_t baseAddr, uint64_t range,
               uint64_t clockFreq)
    : Kernel(device, name, baseAddr, range, std::vector<Register>{}) {
    instancePtr = new XClk_Wiz;
    instancePtr->Config.BaseAddr = baseAddr;
    instancePtr->Config.PrimInClkFreq = 100000000;
    instancePtr->MinErr = 50000;
    this->clockFrequency = clockFreq;
}

uint64_t ClkWiz::getVco() {
    uint32_t Div;
    uint64_t Fvco;
    uint32_t Edge;
    uint32_t Low;
    uint32_t High;
    uint32_t Mult;
    uint32_t reg = read(XCLK_WIZ_REG1_OFFSET);
    Edge = !!(reg & XCLK_WIZ_REG1_EDGE_MASK);
    reg = read(XCLK_WIZ_REG2_OFFSET);
    Low = reg & XCLK_WIZ_CLKFBOUT_L_MASK;
    High = (reg & XCLK_WIZ_CLKFBOUT_H_MASK) >> XCLK_WIZ_CLKFBOUT_H_SHIFT;
    Mult = Low + High + Edge;
    reg = read(XCLK_WIZ_REG13_OFFSET);
    Low = reg & XCLK_WIZ_CLKFBOUT_L_MASK;
    High = (reg & XCLK_WIZ_CLKFBOUT_H_MASK) >> XCLK_WIZ_CLKFBOUT_H_SHIFT;
    reg = read(XCLK_WIZ_REG12_OFFSET);
    Edge = !!(reg & XCLK_WIZ_EDGE_MASK);
    Div = Low + High + Edge;
    if (!Mult) Mult = 1;
    if (!Div) Div = 1;
    Fvco = instancePtr->Config.PrimInClkFreq * Mult / Div;
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "VCO value is: {}", Fvco);
    return Fvco;
}

uint64_t ClkWiz::getClockRate() {
    uint32_t Reg;
    uint32_t Leaf;
    uint64_t Fvco;
    uint64_t Freq;
    uint32_t RegisterOffset;
    uint32_t Edge;
    uint32_t Low;
    uint32_t High;
    uint32_t DivO;
    uint32_t P5en;
    uint32_t Prediv;

    Fvco = getVco();
    Freq = instancePtr->Config.PrimInClkFreq * instancePtr->MVal / instancePtr->DVal /
           instancePtr->OVal;
    return Freq;
}

void ClkWiz::calculateDivisorsHz(uint64_t SetRate) {
    uint32_t m;
    uint32_t d;
    uint32_t Div;
    uint64_t Fvco;
    uint64_t Freq;
    uint64_t Diff;
    uint64_t Minerr = instancePtr->MinErr;
    uint64_t VcoMin;
    uint64_t VcoMax;
    uint32_t Platform;
    uint32_t Mmin;
    uint32_t Mmax;
    uint32_t Dmin;
    uint32_t Dmax;
    uint32_t Omin;
    uint32_t Omax;

    VcoMin = XCLK_VCO_MIN;
    VcoMax = XCLK_VCO_MAX;
    Mmin = XCLK_M_MIN;
    Mmax = XCLK_M_MAX;
    Dmin = XCLK_D_MIN;
    Dmax = XCLK_D_MAX;
    Omin = XCLK_O_MIN;
    Omax = XCLK_O_MAX;
    for (m = Mmin; m <= Mmax; m++) {
        for (d = Dmin; d <= Dmax; d++) {
            Fvco = instancePtr->Config.PrimInClkFreq * m / d;
            if (Fvco >= VcoMin * XCLK_MHZ && Fvco <= VcoMax * XCLK_MHZ) {
                for (Div = Omin; Div <= Omax; Div++) {
                    Freq = Fvco / Div;

                    if (Freq > SetRate) {
                        Diff = Freq - SetRate;
                    } else {
                        Diff = SetRate - Freq;
                    }
                    if (Diff < Minerr) {
                        instancePtr->MVal = m;
                        instancePtr->DVal = d;
                        instancePtr->OVal = Div;
                        utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                                           "M: {}, D: {}, O: {}", m, d, Div);
                        return;
                    }
                }
            }
        }
    }
}

void ClkWiz::updateO() {
    uint32_t HighTime;
    uint32_t DivEdge;
    uint32_t Reg;
    uint32_t P5Enable;
    uint32_t P5fEdge;
    uint32_t RegisterOffset;

    if (instancePtr->OVal > XCLK_O_MAX) {
        instancePtr->OVal = XCLK_O_MAX;
    }
    RegisterOffset =
        XCLK_WIZ_REG3_OFFSET + 0 * 8;  // indicate clock id, needs to be tested to see what value
    HighTime = (instancePtr->OVal / 4);
    Reg = XCLK_WIZ_REG3_PREDIV2 | XCLK_WIZ_REG3_USED | XCLK_WIZ_REG3_MX;
    if (instancePtr->OVal % 4 <= 1) {
        DivEdge = 0;
    } else {
        DivEdge = 1;
    }
    Reg |= (DivEdge << 8);
    P5fEdge = instancePtr->OVal % 2;
    P5Enable = instancePtr->OVal % 2;
    Reg = Reg | P5Enable << XCLK_WIZ_CLKOUT0_P5EN_SHIFT | P5fEdge << XCLK_WIZ_CLKOUT0_P5FEDGE_SHIFT;
    write(RegisterOffset, Reg);
    Reg = HighTime | HighTime << 8;
    RegisterOffset = RegisterOffset + 4;
    write(RegisterOffset, Reg);
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "O value is: {}",
                       instancePtr->OVal);
}

void ClkWiz::updateD() {
    uint32_t HighTime;
    uint32_t DivEdge;
    uint32_t Reg;

    HighTime = (instancePtr->DVal / 2);
    Reg = 0;
    Reg = Reg & ~(1 << XCLK_WIZ_REG12_EDGE_SHIFT);
    DivEdge = instancePtr->DVal % 2;
    Reg = Reg | DivEdge << XCLK_WIZ_REG12_EDGE_SHIFT;
    write(XCLK_WIZ_REG12_OFFSET, Reg);
    Reg = HighTime | HighTime << 8;
    write(XCLK_WIZ_REG13_OFFSET, Reg);
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "D value is: {}",
                       instancePtr->DVal);
}

void ClkWiz::updateM() {
    uint32_t HighTime;
    uint32_t DivEdge;
    uint32_t Reg;
    write(XCLK_WIZ_REG25_OFFSET, 0);

    DivEdge = instancePtr->MVal % 2;
    HighTime = instancePtr->MVal / 2;
    Reg = HighTime | HighTime << 8;
    write(XCLK_WIZ_REG2_OFFSET, Reg);
    Reg = XCLK_WIZ_REG1_PREDIV2 | XCLK_WIZ_REG1_EN | XCLK_WIZ_REG1_MX;

    if (DivEdge) {
        Reg = Reg | (1 << XCLK_WIZ_REG1_EDGE_SHIFT);
    } else {
        Reg = Reg & ~(1 << XCLK_WIZ_REG1_EDGE_SHIFT);
    }
    write(XCLK_WIZ_REG1_OFFSET, Reg);
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "M value is: {}",
                       instancePtr->MVal);
}

uint32_t ClkWiz::waitForLock() {
    uint32_t count = 0;
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "Waiting for clock lock");
    while (!read(XCLK_WIZ_REG4_OFFSET) & 1) {
        if (count == 1000) {
            utils::Logger::log(
                utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                "Error: Timeout waiting for clock lock. Probably values not set correctly");
            utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__,
                               "Trying default frequency, 200MHz");
        }
        usleep(100);
        count++;
    }
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__, "Clock locked");
    return 0;
}

void ClkWiz::setRateHzInternal(uint64_t rate) {
    uint32_t Reg;
    calculateDivisorsHz(rate);
    updateO();
    updateD();
    updateM();
    // weird values no one knows where they come from :)
    write(XCLK_WIZ_REG11_OFFSET, 0x2e);
    write(XCLK_WIZ_REG14_OFFSET, 0xe80);
    write(XCLK_WIZ_REG15_OFFSET, 0x4271);
    write(XCLK_WIZ_REG16_OFFSET, 0x43e9);
    write(XCLK_WIZ_REG17_OFFSET, 0x001C);
    write(XCLK_WIZ_REG26_OFFSET, 0x0001);
}
void ClkWiz::setRateHz(uint64_t rate_, bool verbose) {
    if (verbose)
        utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Setting clock at: {} MHz",
                           std::to_string((double)rate_ / 1000000.0f));

    // start dynamic reconfig
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                       "Starting dynamic reconfiguration");
    write(XCLK_WIZ_REG25_OFFSET, 0);
    setRateHzInternal(rate_);
    write(XCLK_WIZ_RECONFIG_OFFSET, (XCLK_WIZ_RECONFIG_LOAD | XCLK_WIZ_RECONFIG_SADDR));
    uint32_t status = waitForLock();
    if (status != 0) {
        uint32_t reg = read(XCLK_WIZ_REG4_OFFSET);
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Error: Clock not locked : {x}", reg);
        throw std::runtime_error("Clock not locked");
    }
    uint64_t rate = getClockRate();
    if (verbose)
        utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__,
                           "User clock frequency set at: {} MHz",
                           std::to_string((double)rate / 1000000.0f));
}
}  // namespace vrt