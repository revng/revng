#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace Pipeline {

/// A step is a list of pipes that must be exectuted entirelly or not at all.
/// Furthermore a step has a set of containers associated to it as well that
/// will contain the element used for perform the computations.
class Step {
private:
  std::string Name;
  ContainerSet Containers;
  std::vector<PipeWrapper> Pipes;
  Step *PreviousStep;

public:
  /// Clones the Targets from the backing containers of this step
  /// and excutes all the pipes in sequence contained by this step
  /// and returns the transformed containers.
  ///
  /// The contained values stays unchanged.
  ContainerSet cloneAndRun(Context &Ctx,
                           const ContainerToTargetsMap &Targets,
                           llvm::raw_ostream *OS = nullptr);

  /// Returns the set of goals that are already contained in the backing
  /// containers of this step, futhermore adds to the container ToLoad those
  /// that were not present.
  ContainerToTargetsMap
  analyzeGoals(const ContainerToTargetsMap &RequiredGoals,
               ContainerToTargetsMap &AlreadyAvbiable) const;

  /// Returns the predicted state of the Input containers status after the
  /// execution of all the pipes in this step
  ContainerToTargetsMap deduceResults(ContainerToTargetsMap Input) const;

  const ContainerSet &containers() const { return Containers; }

  ContainerSet &containers() { return Containers; }

  void addPipe(PipeWrapper Wrapper) { Pipes.push_back(std::move(Wrapper)); }

  template<typename... PipeWrapperTypes>
  Step(std::string Name,
       ContainerSet Containers,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(nullptr) {}

  template<typename... PipeWrapperTypes>
  Step(std::string Name,
       ContainerSet Containers,
       Step &PreviousStep,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(&PreviousStep) {}

  llvm::StringRef getName() const { return Name; }

  template<typename OStream>
  void dump(OStream &OS, size_t Indents = 0) const {
    indent(OS, Indents);
    OS << "Step " << Name << ":\n";

    indent(OS, Indents + 1);
    OS << "Pipes: \n";
    for (const auto &Pipe : Pipes)
      Pipe.dump(OS, Indents + 2);

    indent(OS, Indents + 1);
    OS << " containers: \n";
    Containers.dump(OS, Indents + 2);
  }

  bool hasPredecessor() const { return PreviousStep != nullptr; }

  const Step &getPredecessor() const {
    revng_assert(PreviousStep != nullptr);
    return *PreviousStep;
  }

  Step &getPredecessor() {
    revng_assert(PreviousStep != nullptr);
    return *PreviousStep;
  }

  /// Drops from the backing containers all the targets presents in containers
  /// status
  llvm::Error invalidate(const ContainerToTargetsMap &ToRemove);

  void dump() const debug_function { dump(dbg); }

  llvm::Error storeToDisk(llvm::StringRef DirPath) const;
  llvm::Error loadFromDisk(llvm::StringRef DirPath);

private:
  static void removeSatisfiedGoals(TargetsList &RequiredInputs,
                                   const ContainerBase &CachedSymbols,
                                   TargetsList &ToLoad);

  void removeSatisfiedGoals(ContainerToTargetsMap &Targets,
                            ContainerToTargetsMap &ToLoad) const;

  void explainExecutedPipe(const PipeWrapper &Wrapper,
                           llvm::raw_ostream *OS,
                           size_t Indents = 0) const;
  void explainStartStep(const ContainerToTargetsMap &Wrapper,
                        llvm::raw_ostream *OS,
                        size_t Indents = 0) const;
};

} // namespace Pipeline
