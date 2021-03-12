python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/DeadRegisterArgumentsOfFunction.dot > lib/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunction.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/DeadReturnValuesOfFunctionCall.dot > lib/ABIAnalyses/Generated/DeadReturnValuesOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/UsedArgumentsOfFunction.dot > lib/ABIAnalyses/Generated/UsedArgumentsOfFunction.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/UsedReturnValuesOfFunctionCall.dot > lib/ABIAnalyses/Generated/UsedReturnValuesOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/RegisterArgumentsOfFunctionCall.dot > lib/ABIAnalyses/Generated/RegisterArgumentsOfFunctionCall.h
python3 scripts/monotone-framework-lattice.py lib/ABIAnalyses/ABIAnalysis.template lib/ABIAnalyses/UsedReturnValuesOfFunction.dot > lib/ABIAnalyses/Generated/UsedReturnValuesOfFunction.h
