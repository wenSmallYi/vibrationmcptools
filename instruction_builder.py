class MCPInstructionBuilder:
    @staticmethod
    def build(semantics_list: list) -> list:
        instrs = []
        for semantics in semantics_list:
            instr = {
                "axis": semantics.get("axis", "Z"),
                "features": semantics.get("features", ["RMS", "Skewness", "Kurtosis", "CrestFactor", "Estimated Speed"]),
                "processing": semantics.get("processing", {})
            }
            instrs.append(instr)
        return instrs
