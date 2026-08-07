//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Analyses/ConvertFunctionsToCABI.h"
#include "revng/ABI/Analyses/ConvertFunctionsToRaw.h"
#include "revng/Canonicalize/FixPointerSize.h"
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
#include "revng/EditCBody/EditCBodyAnalysis.h"
#include "revng/EditCType/EditCTypeAnalysis.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/InlineAlwaysInlineFunctions.h"
#include "revng/FunctionIsolation/InvokeIsolatedFunctions.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/FunctionIsolation/PromoteCSVs.h"
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
#include "revng/RemoveLiftingArtifacts/PromoteInitCSVToUndef.h"
#include "revng/RemoveLiftingArtifacts/RemoveLiftingArtifacts.h"
#include "revng/SegmentReferences/DetectCStrings.h"
#include "revng/SegmentReferences/EmitFunctionPointers.h"
#include "revng/SegmentReferences/EmitSegmentReferences.h"
#include "revng/SegmentReferences/EmitStringConstants.h"
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

REGISTER(Container, AssemblyContainer);
REGISTER(Container, AssemblyInternalContainer);
REGISTER(Container, BinariesContainer);
REGISTER(Container, CallGraphContainer);
REGISTER(Container, CallGraphSliceContainer);
REGISTER(Container, CFGMap);
REGISTER(Container, CliftFunctionContainer);
REGISTER(Container, CliftModuleContainer);
REGISTER(Container, CliftSingleTypeContainer);
REGISTER(Container, CrossRelationsContainer);
REGISTER(Container, FunctionControlFlowContainer);
REGISTER(Container, HexDumpContainer);
REGISTER(Container, LLVMFunctionContainer);
REGISTER(Container, LLVMRootContainer);
REGISTER(Container, ObjectFileContainer);
REGISTER(Container, PTMLCContainer);
REGISTER(Container, PTMLCFunctionContainer);
REGISTER(Container, CTypeContainer);
REGISTER(Container, RecompilableArchiveContainer);
REGISTER(Container, TranslatedContainer);

//
// Pipes
//

using namespace revng::pypeline::pipes;

REGISTER(Pipe, FixPointerSize);
REGISTER(Pipe, PureLLVMPassesPipe);
REGISTER(Pipe, PureLLVMPassesRootPipe);
REGISTER(Pipe, PureMLIRPassesPipe);

using namespace revng::pypeline::piperuns;

REGISTER(FunctionPipeRun, AttachDebugInfo);
REGISTER(FunctionPipeRun, Clifter);
REGISTER(FunctionPipeRun, CollectCFG);
REGISTER(FunctionPipeRun, EmitC);
REGISTER(FunctionPipeRun, EmitFunctionPointers);
REGISTER(FunctionPipeRun, EmitStringConstants);
REGISTER(FunctionPipeRun, EnforceABI);
REGISTER(FunctionPipeRun, ImportDescriptiveFunctionInfo);
REGISTER(FunctionPipeRun, ImportFunctionDataModel);
REGISTER(FunctionPipeRun, InlineAlwaysInlineFunctions);
REGISTER(FunctionPipeRun, InjectStackSizeProbesAtCallSites);
REGISTER(FunctionPipeRun, Isolate);
REGISTER(FunctionPipeRun, ProcessAssembly);
REGISTER(FunctionPipeRun, PromoteCSVs);
REGISTER(FunctionPipeRun, PromoteInitCSVToUndef);
REGISTER(FunctionPipeRun, PromoteStackPointer);
REGISTER(FunctionPipeRun, RemoveLiftingArtifacts);
REGISTER(FunctionPipeRun, SegregateStackAccesses);
REGISTER(FunctionPipeRun, SimplifySwitch);
REGISTER(FunctionPipeRun, SwitchToStatements);
REGISTER(FunctionPipeRun, VerifyFunctionAgainstModel);
REGISTER(FunctionPipeRun, YieldAssembly);
REGISTER(FunctionPipeRun, YieldCallGraphSlice);
REGISTER(FunctionPipeRun, YieldCFG);

REGISTER(SingleOutputPipeRun, CompileRootModule);
REGISTER(SingleOutputPipeRun, EmitCAsDirectory);
REGISTER(SingleOutputPipeRun, EmitCAsSingleFile);
REGISTER(SingleOutputPipeRun, EmitHelperHeader);
REGISTER(SingleOutputPipeRun, EmitSegmentReferences);
REGISTER(SingleOutputPipeRun, EmitTypeAndGlobalHeader);
REGISTER(SingleOutputPipeRun, HexDump);
REGISTER(SingleOutputPipeRun, ImportDescriptiveInfo);
REGISTER(SingleOutputPipeRun, ImportFunctionDeclarations);
REGISTER(SingleOutputPipeRun, ImportSegmentDeclarations);
REGISTER(SingleOutputPipeRun, ImportTypes);
REGISTER(SingleOutputPipeRun, InvokeIsolatedFunctions);
REGISTER(SingleOutputPipeRun, Lift);
REGISTER(SingleOutputPipeRun, LinkForTranslation);
REGISTER(SingleOutputPipeRun, LinkSupport);
REGISTER(SingleOutputPipeRun, MergeLLVMModules);
REGISTER(SingleOutputPipeRun, ProcessCallGraph);
REGISTER(SingleOutputPipeRun, VerifyAgainstModel);
REGISTER(SingleOutputPipeRun, YieldCallGraph);

REGISTER(TypeDefinitionPipeRun, EmitSingleTypeDefinition);

//
// Analyses
//

using namespace revng::pypeline::analyses;

REGISTER(Analysis, AnalyzeDataLayout);
REGISTER(Analysis, ApplyDiff);
REGISTER(Analysis, ConvertFunctionsToCABI);
REGISTER(Analysis, ConvertFunctionsToRaw);
REGISTER(Analysis, DetectABI);
REGISTER(Analysis, revng::pypeline::analyses::DetectCStrings);
REGISTER(Analysis, DetectStackSize);
REGISTER(Analysis, EditCBody);
REGISTER(Analysis, EditCType);
REGISTER(Analysis, ImportPrototypesFromDatabase);
REGISTER(Analysis, LLMRename);
REGISTER(Analysis, ParseBinaryAnalysis);
REGISTER(Analysis, SetModel);
REGISTER(Analysis, VerifyDiff);
REGISTER(Analysis, VerifyModel);
