#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Support/MetaAddress.h"

void printSingleCFile(llvm::raw_ostream &Out,
                      const revng::pipes::FunctionStringMap &Functions,
                      const std::set<MetaAddress> &Targets);
