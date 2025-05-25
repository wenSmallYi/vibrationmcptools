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
                "params": {"low": 1000, "high": 3500}
            },
            "features": ["RMS", "Skewness", "Kurtosis", "CrestFactor"]
        }

    @staticmethod
    def bandpass_filter(data: np.ndarray, fs: int, lowcut: float, highcut: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, data)

    @staticmethod
    def lowpass_filter(data: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(4, cutoff / nyq, btype="low")
        return filtfilt(b, a, data)

    @staticmethod
    def highpass_filter(data: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(4, cutoff / nyq, btype="high")
        return filtfilt(b, a, data)

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return data
        return 2 * (data - min_val) / (max_val - min_val) - 1

    @staticmethod
    def calculate_features(data: np.ndarray, features: list) -> dict:
        result = {}

        # 定義所有支援的特徵對應公式
        feature_map = {
            "RMS": lambda d: float(np.sqrt(np.mean(d ** 2))),
            "StandardDeviation": lambda d: float(np.std(d, ddof=1)),  # 樣本標準差
            "Kurtosis": lambda d: float(kurtosis(d)),
            "Skewness": lambda d: float(skew(d)),
            "Crest Factor": lambda d: float(np.max(np.abs(d)) / np.sqrt(np.mean(d ** 2))) if np.sqrt(np.mean(d ** 2)) != 0 else 0,
            "MAX": lambda d: float(np.max(d)),
            "MIN": lambda d: float(np.min(d)),
            "MEAN": lambda d: float(np.mean(d)),
            "MEDIAN": lambda d: float(np.median(d)),
            # 這裡可以自行擴充更多
        }

        for feat in features:
            # 忽略大小寫差異
            key = feat.upper() if feat.isupper() else feat
            # 兼容用戶可能輸入的小寫
            for k, func in feature_map.items():
                if feat.lower() == k.lower():
                    try:
                        result[k] = func(data)
                    except Exception as e:
                        result[k] = f"計算錯誤: {str(e)}"

        return result

    @staticmethod
    def calculate_frequency_features(data: np.ndarray, fs: int, blade_count: int = 6) -> dict:
        freqs, pxx = welch(data, fs, nperseg=fs // 2)
        peak_idx = np.argmax(pxx)
        peak_freq = freqs[peak_idx]
        rpm = peak_freq * 60 / blade_count

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
    def analyze_with_instructions(filepath: str, fs: int, instructions: list) -> list:
        """
        支援多組指令與多軸的批次分析

        instructions: List[dict]，
        例如:
        [
          {"axis": ["X", "Y"], "processing": {...}, "features": [...]},
          {"axis": "Z", "processing": {...}, "features": [...]},
          ...
        ]
        """
        df = pd.read_csv(filepath)
        results = []
        for instr in instructions:
            axes = instr.get("axis", "Z")
            if isinstance(axes, str):
                axes = [axes]
            for axis in axes:
                if axis not in df.columns:
                    results.append({"axis": axis, "error": f"Axis '{axis}' not found in data"})
                    continue
                data = df[axis].to_numpy()[:fs]
                data -= np.mean(data)
                data = Tools.normalize(data)

                # 處理濾波方法
                proc = instr.get("processing", {})
                method = proc.get("method", "")
                params = proc.get("params", {})
                if method == "bandpass_filter":
                    data = Tools.bandpass_filter(data, fs, params.get("low", 1000), params.get("high", 3500))
                elif method == "lowpass_filter":
                    data = Tools.lowpass_filter(data, fs, params.get("cutoff", 3000))
                elif method == "highpass_filter":
                    data = Tools.highpass_filter(data, fs, params.get("cutoff", 1000))
                # 可以繼續擴充

                features = instr.get("features", [])
                result = Tools.calculate_features(data, features)
                if any(f in features for f in [
                    "Peak Frequency (Hz)", "Estimated Speed", "Peak PSD", "Spectrum Skewness", "Spectrum Kurtosis"
                ]):
                    result.update(Tools.calculate_frequency_features(data, fs))
                result["axis"] = axis
                result["method"] = method
                results.append(result)
        return results
