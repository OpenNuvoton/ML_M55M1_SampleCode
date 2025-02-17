/**************************************************************************//**
 * @file     ModelBinParser.hpp
 * @version  V1.00
 * @brief    Parser model binary function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __MODEL_BIN_PARSER_HPP__
#define __MODEL_BIN_PARSER_HPP__

#include <cinttypes>
#include <vector>
#include <string>

/**
  * @brief Initiate model binary parser resources
  * @return 0: Success, <0: Fail
  * \hideinitializer
  */

int ModelBinParser_Init(char *pu8ModelBin);
int ModelBinParser_GetTfliteOffset(void);
int ModelBinParser_GetTfliteSize(void);
int ModelBinParser_GetLabels(std::vector<std::string> &labels);

#endif
