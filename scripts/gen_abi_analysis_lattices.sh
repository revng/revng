# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/DeadRegisterArgumentsOfFunction.dot > include/revng/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunctionLattice.h
# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/DeadReturnValuesOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/DeadReturnValuesOfFunctionCallLattice.h
# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/RegisterArgumentsOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/RegisterArgumentsOfFunctionCallLattice.h
# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedArgumentsOfFunction.dot > include/revng/ABIAnalyses/Generated/UsedArgumentsOfFunctionLattice.h
# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedReturnValuesOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/UsedReturnValuesOfFunctionCallLattice.h
# python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedReturnValuesOfFunction.dot > include/revng/ABIAnalyses/Generated/UsedReturnValuesOfFunctionLattice.h

python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/DeadRegisterArgumentsOfFunction.dot > include/revng/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunction.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/DeadReturnValuesOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/DeadReturnValuesOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedArgumentsOfFunction.dot > include/revng/ABIAnalyses/Generated/UsedArgumentsOfFunction.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedReturnValuesOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/UsedReturnValuesOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/RegisterArgumentsOfFunctionCall.dot > include/revng/ABIAnalyses/Generated/RegisterArgumentsOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/StackAnalysis/UsedReturnValuesOfFunction.dot > include/revng/ABIAnalyses/Generated/UsedReturnValuesOfFunction.h
