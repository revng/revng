/// \file Kinds.cpp
/// \brief contains the various kinds declared by revng

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/IsolatedKind.h"
#include "revng/Pipes/Kinds.h"

using namespace std;
using namespace pipeline;
using namespace revng::pipes;

Rank revng::pipes::RootRank("Root Rank");

Kind revng::pipes::Binary("Binary", &RootRank);

Kind revng::pipes::Object("Object", &RootRank);
Kind revng::pipes::Translated("Translated", &RootRank);
Kind revng::pipes::Dead("Dead", &RootRank);

Kind revng::pipes::CFepper("CFepper", &FunctionsRank);
Kind revng::pipes::ABIEnforced("ABIEnforced", Isolated, &FunctionsRank);

static RegisterKind K1(Translated);
static RegisterKind K2(Binary);
static RegisterKind K3(Dead);
static RegisterKind K4(Object);
static RegisterKind K5(ABIEnforced);
static RegisterKind K6(CFepper);
