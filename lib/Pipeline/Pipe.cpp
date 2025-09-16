/// \file Pipe.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Pipe.h"

using namespace pipeline;

using llvm::StringRef;

void InvalidationMetadata::registerTargetsDependingOn(const Context &Context,
                                                      StringRef GlobalName,
                                                      const TupleTreePath &Path,
                                                      ContainerToTargetsMap
                                                        &Out,
                                                      Logger<> &Log) const {
  if (auto Iter = PathCache.find(GlobalName); Iter != PathCache.end()) {
#ifdef VECTOR_BIMAP
    auto &Bimap = Iter->second;
    auto Range = Bimap.find(Path);

    for (const auto &Entry : Range) {
      revng_log(Log,
                Entry.second.getTarget().toString()
                  << " in " << Entry.second.getContainerName() << "\n");
      Out.add(Entry.second.getContainerName(), Entry.second.getTarget());
    }
#else
    auto &Bimap = Iter->second;
    auto It = Bimap.find(Path);
    if (It == Bimap.end())
      return;

    if (Log.isEnabled()) {
      Log << "Registering: ";
      for (const auto &Entry : It->second) {
        Log << Entry.getTarget().toString() << " in "
            << Entry.getContainerName() << "\n";
      }
      Log << DoLog;
    }

    for (const auto &Entry : It->second)
      Out.add(Entry.getContainerName(), Entry.getTarget());
#endif
  }
}

void InvalidationMetadata::remove(const ContainerToTargetsMap &Map) {
  for (auto &Pair : Map) {
    auto Iter = PathCache.find(Pair.first());
    if (Iter == PathCache.end())
      continue;

    Iter->second.remove(Pair.second, Pair.first());
  }
}

void InvalidationMetadata::dump(const pipeline::Context &Context,
                                unsigned Indentation) const {
#if 0
  for (const auto &[GlobalName, InvalidationData] : PathCache) {
    indent(dbg, Indentation);
    dbg << "Global " << GlobalName.str() << ":\n";

    for (const auto &[Path, Targets] : PathCache.find(GlobalName)->second) {
      indent(dbg, Indentation + 1);

      dbg << llvm::cantFail(Context.getGlobals().get(GlobalName))
               ->serializePath(Path)
               .value_or("(unavailable)")
          << ":\n";

      for (const TargetInContainer &Target : Targets) {
        Target.dump(dbg, Indentation + 2);
      }
    }
  }
#endif
}
