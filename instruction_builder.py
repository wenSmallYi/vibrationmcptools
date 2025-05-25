class MCPInstructionBuilder:
    @staticmethod
    def build(semantics: dict) -> dict:
        instr = {
            "axis": semantics.get("axis", "Y"),
            "features": semantics.get("features", ["RMS", "Kurtosis"]),
            "processing": {}
        }
        if "low_freq" in semantics and "high_freq" in semantics:
            instr["processing"] = {
                "method": "bandpass_filter",
                "params": {
                    "low": semantics["low_freq"],
                    "high": semantics["high_freq"]
                }
            }
        return instr