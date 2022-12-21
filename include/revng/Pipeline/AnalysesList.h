#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "revng/Pipeline/Analysis.h"

namespace pipeline {

class AnalysisReference {
private:
  std::string StepName;
  std::string AnalysisName;

public:
  AnalysisReference(std::string StepName, std::string AnalysisName) :
    StepName(StepName), AnalysisName(AnalysisName) {}

  llvm::StringRef getStepName() const { return StepName; }
  llvm::StringRef getAnalysisName() const { return AnalysisName; }
};

class AnalysesList {
private:
  std::string Name;
  using Container = std::vector<AnalysisReference>;
  Container Analyses;

public:
  using iterator = Container::iterator;
  using const_iterator = Container::const_iterator;

public:
  AnalysesList(llvm::StringRef Name,
               llvm::ArrayRef<AnalysisReference> Analyses = {}) :
    Name(Name.str()), Analyses(Analyses) {}

  iterator begin() { return Analyses.begin(); }
  iterator end() { return Analyses.end(); }
  const_iterator begin() const { return Analyses.begin(); }
  const_iterator end() const { return Analyses.end(); }

  void addAnalysisReference(std::string StepName, std::string AnalysisName) {
    Analyses.emplace_back(StepName, AnalysisName);
  }

  size_t size() const { return Analyses.size(); }
  const AnalysisReference &at(size_t Index) const { return Analyses.at(Index); }

  llvm::StringRef getName() const { return Name; }
};

} // namespace pipeline
