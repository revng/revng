//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Analyses/ConvertFunctionsToCABI.h"
#include "revng/ABI/Analyses/ConvertFunctionsToRaw.h"
#include "revng/Canonicalize/SimplifySwitch.h"
#include "revng/Canonicalize/SwitchToStatements.h"
#include "revng/CliftPipes/Clifter.h"
#include "revng/CliftPipes/EmitC.h"
#include "revng/CliftPipes/EmitCAsDirectory.h"
#include "revng/CliftPipes/EmitCAsSingleFile.h"
#include "revng/CliftPipes/Headers.h"
#include "revng/CliftPipes/ImportDataModel.h"
#include "revng/CliftPipes/ImportDescriptiveInfo.h"
#include "revng/CliftPipes/ImportTypes.h"
#include "revng/CliftPipes/VerifyAgainstModel.h"
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
#include "revng/ImportFromC/ImportFromCAnalysis.h"
#include "revng/LLMRename/LLMRenameAnalysis.h"
#include "revng/Lift/Lift.h"
#include "revng/Lift/LinkSupportPipe.h"
#include "revng/Model/Importer/Binary/ImportBinaryAnalysis.h"
#include "revng/Model/Importer/ImportPrototypesFromDatabase.h"
#include "revng/Pipebox/LLVMPipe.h"
#include "revng/Pipebox/MLIRPipe.h"
#include "revng/Pipebox/MergeLLVMModules.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/Registrars.h"
#include "revng/PipeboxCommon/ModelManipulationAnalyses.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/PromoteStackPointer/DetectStackSize.h"
#include "revng/PromoteStackPointer/InjectStackSizeProbesAtCallSites.h"
#include "revng/PromoteStackPointer/PromoteStackPointer.h"
#include "revng/PromoteStackPointer/SegregateStackAccesses.h"
#include "revng/Recompile/CompileModulePipe.h"
#include "revng/Recompile/LinkForTranslationPipe.h"
#include "revng/RemoveLiftingArtifacts/CleanupIR.h"
#include "revng/RemoveLiftingArtifacts/MakeSegmentRef.h"
#include "revng/RemoveLiftingArtifacts/PromoteInitCSVToUndef.h"
#include "revng/RemoveLiftingArtifacts/RemoveLiftingArtifacts.h"
#include "revng/Yield/HexDump.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"
#include "revng/Yield/Pipes/YieldAssembly.h"
#include "revng/Yield/Pipes/YieldCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraphSlice.h"

#define REGISTER(TYPE, NAME) \
  static Register##TYPE<NAME> CONCAT3(TYPE, _, __COUNTER__)

//
// Containers
//

using namespace revng::pypeline;

REGISTER(Container, LLVMRootContainer);
REGISTER(Container, LLVMFunctionContainer);
REGISTER(Container, PTMLCBytesContainer);
REGISTER(Container, BinariesContainer);
REGISTER(Container, PTMLCTypeContainer);
REGISTER(Container, CFGMap);
REGISTER(Container, HexDumpContainer);
REGISTER(Container, AssemblyInternalContainer);
REGISTER(Container, AssemblyContainer);
REGISTER(Container, ObjectFileContainer);
REGISTER(Container, TranslatedContainer);
REGISTER(Container, CliftFunctionContainer);
REGISTER(Container, PTMLCFunctionBytesContainer);
REGISTER(Container, CrossRelationsContainer);
REGISTER(Container, CallGraphContainer);
REGISTER(Container, CallGraphSliceContainer);
REGISTER(Container, FunctionControlFlowContainer);
REGISTER(Container, CliftModuleContainer);
REGISTER(Container, CliftSingleTypeContainer);
REGISTER(Container, PTMLCTypeBytesContainer);
REGISTER(Container, RecompilableArchiveContainer);

//
// Pipes
//

using namespace revng::pypeline::pipes;
using namespace revng::pypeline::piperuns;

REGISTER(SingleOutputPipeRun, Lift);
REGISTER(Pipe, PureLLVMPassesRootPipe);
REGISTER(Pipe, PureLLVMPassesPipe);
REGISTER(SingleOutputPipeRun, ModelToHeader);
REGISTER(TypeDefinitionPipeRun, GenerateModelTypeDefinition);
REGISTER(FunctionPipeRun, CollectCFG);
REGISTER(FunctionPipeRun, Isolate);
REGISTER(FunctionPipeRun, AttachDebugInfo);
REGISTER(FunctionPipeRun, EnforceABI);
REGISTER(FunctionPipeRun, PromoteCSVs);
REGISTER(SingleOutputPipeRun, HexDump);
REGISTER(FunctionPipeRun, ProcessAssembly);
REGISTER(FunctionPipeRun, YieldAssembly);
REGISTER(SingleOutputPipeRun, LinkSupport);
REGISTER(SingleOutputPipeRun, CompileRootModule);
REGISTER(SingleOutputPipeRun, LinkForTranslation);
REGISTER(SingleOutputPipeRun, InvokeIsolatedFunctions);
REGISTER(FunctionPipeRun, RemoveLiftingArtifacts);
REGISTER(FunctionPipeRun, PromoteInitCSVToUndef);
REGISTER(FunctionPipeRun, InjectStackSizeProbesAtCallSites);
REGISTER(FunctionPipeRun, PromoteStackPointer);
REGISTER(FunctionPipeRun, SimplifySwitch);
REGISTER(FunctionPipeRun, LegacySegregateStackAccesses);
REGISTER(FunctionPipeRun, MakeSegmentRef);
REGISTER(FunctionPipeRun, SegregateStackAccesses);
REGISTER(FunctionPipeRun, SwitchToStatements);
REGISTER(FunctionPipeRun, Clifter);
REGISTER(Pipe, PureMLIRPassesPipe);
REGISTER(SingleOutputPipeRun, VerifyAgainstModel);
REGISTER(SingleOutputPipeRun, ImportDescriptiveInfo);
REGISTER(FunctionPipeRun, EmitC);
REGISTER(SingleOutputPipeRun, EmitCAsSingleFile);
REGISTER(SingleOutputPipeRun, MergeLLVMModules);
REGISTER(SingleOutputPipeRun, CleanupIR);
REGISTER(SingleOutputPipeRun, ProcessCallGraph);
REGISTER(SingleOutputPipeRun, YieldCallGraph);
REGISTER(FunctionPipeRun, YieldCallGraphSlice);
REGISTER(FunctionPipeRun, YieldCFG);
REGISTER(FunctionPipeRun, ImportDescriptiveFunctionInfo);
REGISTER(FunctionPipeRun, VerifyFunctionAgainstModel);
REGISTER(FunctionPipeRun, ImportFunctionDataModel);
REGISTER(SingleOutputPipeRun, ImportTypes);
REGISTER(SingleOutputPipeRun, ImportFunctionDeclarations);
REGISTER(SingleOutputPipeRun, ImportSegmentDeclarations);
REGISTER(SingleOutputPipeRun, EmitTypeAndGlobalHeader);
REGISTER(SingleOutputPipeRun, EmitHelperHeader);
REGISTER(TypeDefinitionPipeRun, EmitSingleTypeDefinition);
REGISTER(SingleOutputPipeRun, EmitCAsDirectory);

//
// Analyses
//

using namespace revng::pypeline::analyses;

REGISTER(Analysis, ApplyDiff);
REGISTER(Analysis, VerifyDiff);
REGISTER(Analysis, SetModel);
REGISTER(Analysis, VerifyModel);
REGISTER(Analysis, ParseBinaryAnalysis);
REGISTER(Analysis, ImportPrototypesFromDatabase);
REGISTER(Analysis, DetectABI);
REGISTER(Analysis, DetectStackSize);
REGISTER(Analysis, AnalyzeDataLayout);
REGISTER(Analysis, ConvertFunctionsToCABI);
REGISTER(Analysis, ConvertFunctionsToRaw);
REGISTER(Analysis, ImportFromC);
REGISTER(Analysis, LLMRename);
