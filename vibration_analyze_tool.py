import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew

class Tools:
    @staticmethod
    def default_instruction() -> dict:
        return {
            "axis": "Z",
            "processing": {
                "method": "bandpass_filter",
                "params": { "low": 1000, "high": 3500 }
            },
            "features": ["RMS", "Skewness", "Kurtosis", "CrestFactor", "Estimated Speed"]
        }

    @staticmethod
    def merge_instruction(default: dict, override: dict) -> dict:
        merged = default.copy()
        if "axis" in override:
            merged["axis"] = override["axis"]
        if "processing" in override:
            merged["processing"] = override["processing"]
        if "features" in override:
            merged["features"] = list(set(default.get("features", [])).union(override["features"]))
        return merged

    @staticmethod
    def bandpass_filter(data: np.ndarray, fs: int, lowcut: float, highcut: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, data)

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        # 將資料正規化至 [-1, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return data
        return 2 * (data - min_val) / (max_val - min_val) - 1

    @staticmethod
    def calculate_features(data: np.ndarray, features: list) -> dict:
        result = {}
        if "RMS" in features:
            result["RMS"] = float(np.sqrt(np.mean(data ** 2)))
        if "Kurtosis" in features:
            result["Kurtosis"] = float(kurtosis(data))
        if "Skewness" in features:
            result["Skewness"] = float(skew(data))
        if "Crest Factor" in features:
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data ** 2))
            result["Crest Factor"] = float(peak / rms) if rms != 0 else 0
        return result

    @staticmethod
    def calculate_frequency_features(data: np.ndarray, fs: int, blade_count: int = 6) -> dict:
        from scipy.stats import kurtosis, skew
        from scipy.signal import welch
        import numpy as np

        freqs, pxx = welch(data, fs, nperseg=fs // 2)
        peak_idx = np.argmax(pxx)
        peak_freq = freqs[peak_idx]
        rpm = peak_freq * 60 / blade_count

        # 頻譜統計特徵
        freq_skewness = float(skew(pxx))
        freq_kurtosis = float(kurtosis(pxx))

        return {
            "Peak PSD": float(pxx[peak_idx]),
            "Spectrum Skewness": freq_skewness,
            "Spectrum Kurtosis": freq_kurtosis,
            "Peak Frequency (Hz)": float(peak_freq),
            "Estimated Speed (RPM)": float(rpm),
        }

    @staticmethod
    def analyze_with_instruction(filepath: str, fs: int, instruction: dict) -> dict:
        df = pd.read_csv(filepath)
        axis = instruction.get("axis", "Z")
        if axis not in df.columns:
            return {"error": f"Axis '{axis}' not found in data"}

        data = df[axis].to_numpy()[:fs]
        data -= np.mean(data)
        data = Tools.normalize(data)

        if instruction.get("processing", {}).get("method") == "bandpass_filter":
            params = instruction["processing"]["params"]
            data = Tools.bandpass_filter(data, fs, params["low"], params["high"])

        features = instruction.get("features", [])
        result = Tools.calculate_features(data, features)

        if any(f in features for f in ["Estimated Speed", "Peak PSD", "Peak Frequency (Hz)", "Spectrum Skewness", "Spectrum Kurtosis"]):
            result.update(Tools.calculate_frequency_features(data, fs))

        return result