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

Granularity revng::pipes::RootGranularity("Root Granularity");

Kind revng::pipes::Binary("Binary", &RootGranularity);

Kind revng::pipes::Object("Object", &RootGranularity);
Kind revng::pipes::Translated("Translated", &RootGranularity);
Kind revng::pipes::Dead("Dead", &RootGranularity);

Kind revng::pipes::CFepper("CFepper", &FunctionsGranularity);
Kind revng::pipes::ABIEnforced("ABIEnforced", Isolated, &FunctionsGranularity);

static RegisterKind K1(Translated);
static RegisterKind K2(Binary);
static RegisterKind K3(Dead);
static RegisterKind K4(Object);
static RegisterKind K5(ABIEnforced);
static RegisterKind K6(CFepper);
