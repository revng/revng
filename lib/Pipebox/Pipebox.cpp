//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Analyses/ConvertFunctionsToCABI.h"
#include "revng/Canonicalize/SimplifySwitch.h"
#include "revng/DataLayoutAnalysis/DLA.h"
#include "revng/EarlyFunctionAnalysis/AttachDebugInfo.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/EarlyFunctionAnalysis/DetectABI.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/InvokeIsolatedFunctions.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/FunctionIsolation/PromoteCSVs.h"
#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng/Lift/Lift.h"
#include "revng/Lift/LinkSupportPipe.h"
#include "revng/Model/Importer/Binary/ImportBinaryAnalysis.h"
#include "revng/Model/Importer/WellKnownModels.h"
#include "revng/Pipebox/LLVMPipe.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/CliftContainer.h"
#include "revng/PipeboxCommon/Helpers/Registrars.h"
#include "revng/PipeboxCommon/ModelManipulationAnalyses.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/PromoteStackPointer/DetectStackSize.h"
#include "revng/PromoteStackPointer/InjectStackSizeProbesAtCallSites.h"
#include "revng/PromoteStackPointer/PromoteStackPointer.h"
#include "revng/PromoteStackPointer/SegregateStackAccesses.h"
#include "revng/Recompile/CompileModulePipe.h"
#include "revng/Recompile/LinkForTranslationPipe.h"
#include "revng/RemoveLiftingArtifacts/MakeSegmentRef.h"
#include "revng/RemoveLiftingArtifacts/PromoteInitCSVToUndef.h"
#include "revng/RemoveLiftingArtifacts/RemoveLiftingArtifacts.h"
#include "revng/Yield/HexDump.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldAssembly.h"

using namespace revng::pypeline;

//
// Containers
//

static RegisterContainer<LLVMRootContainer> C1;
static RegisterContainer<LLVMFunctionContainer> C3;
static RegisterContainer<CBytesContainer> C4;
static RegisterContainer<BinariesContainer> C5;
static RegisterContainer<PTMLCTypeContainer> C6;
static RegisterContainer<CFGMap> C7;
static RegisterContainer<HexDumpContainer> C8;
static RegisterContainer<AssemblyInternalContainer> C9;
static RegisterContainer<AssemblyContainer> C10;
static RegisterContainer<ObjectFileContainer> C11;
static RegisterContainer<TranslatedContainer> C12;
static RegisterContainer<CliftFunctionContainer> C13;

//
// Pipes
//

using namespace revng::pypeline::pipes;
using namespace revng::pypeline::piperuns;
namespace piperuns = revng::pypeline::piperuns;

static RegisterSingleOutputPipeRun<Lift> P1;
static RegisterPipe<PureLLVMPassesRootPipe> P2;
static RegisterPipe<PureLLVMPassesPipe> P3;
static RegisterSingleOutputPipeRun<ModelToHeader> P4;
static RegisterTypeDefinitionPipeRun<GenerateModelTypeDefinition> P5;
static RegisterFunctionPipeRun<CollectCFG> P6;
static RegisterFunctionPipeRun<Isolate> P7;
static RegisterFunctionPipeRun<AttachDebugInfo> P8;
static RegisterFunctionPipeRun<piperuns::EnforceABI> P9;
static RegisterFunctionPipeRun<PromoteCSVs> P10;
static RegisterSingleOutputPipeRun<HexDump> P11;
static RegisterFunctionPipeRun<ProcessAssembly> P12;
static RegisterFunctionPipeRun<YieldAssembly> P13;
static RegisterSingleOutputPipeRun<LinkSupport> P14;
static RegisterSingleOutputPipeRun<CompileRootModule> P15;
static RegisterSingleOutputPipeRun<LinkForTranslation> P16;
static RegisterSingleOutputPipeRun<InvokeIsolatedFunctions> P17;
static RegisterFunctionPipeRun<RemoveLiftingArtifacts> P18;
static RegisterFunctionPipeRun<PromoteInitCSVToUndef> P19;
static RegisterFunctionPipeRun<InjectStackSizeProbesAtCallSites> P20;
static RegisterFunctionPipeRun<PromoteStackPointer> P21;
static RegisterFunctionPipeRun<SimplifySwitch> P22;
static RegisterFunctionPipeRun<LegacySegregateStackAccesses> P23;
static RegisterFunctionPipeRun<MakeSegmentRef> P24;

//
// Analyses
//

using namespace revng::pypeline::analyses;

static RegisterAnalysis<ApplyDiff> A1;
static RegisterAnalysis<VerifyDiff> A2;
static RegisterAnalysis<SetModel> A3;
static RegisterAnalysis<VerifyModel> A4;
static RegisterAnalysis<ParseBinaryAnalysis> A5;
static RegisterAnalysis<ImportWellKnownModelsAnalysis> A6;
static RegisterAnalysis<DetectABI> A7;
static RegisterAnalysis<DetectStackSize> A8;
static RegisterAnalysis<AnalyzeDataLayout> A9;
static RegisterAnalysis<ConvertFunctionsToCABI> A10;
