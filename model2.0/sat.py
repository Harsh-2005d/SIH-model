import os
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler


def impute_missing_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Robustly fills NaN pixels in a 3D [time, H, W] array.
    Uses temporal neighbors' mean; falls back gracefully if neighbors are empty.
    """
    arr = arr.copy()
    t, h, w = arr.shape

    # --- Handle first day ---
    if np.isnan(arr[0]).all():
        # find first non-empty day
        for d in range(1, t):
            if not np.isnan(arr[d]).all():
                arr[0] = arr[d]
                break
            else:
                arr[0] = np.zeros((h, w))
    else:
        mean_val = np.nanmean(arr[0])
        arr[0][np.isnan(arr[0])] = mean_val

    # --- Handle rest of the days ---
    for day in range(1, t):
        elem = arr[day]
        missing = np.isnan(elem)

        prev = arr[day - 1]
        after = arr[day + 1] if day + 1 < t else None

        # if both prev and after exist
        if after is not None:
            stack = np.stack([prev, after])
            mean = np.nanmean(stack, axis=0)
        else:
            mean = np.nan_to_num(prev, nan=np.nanmean(prev))

        # fallback: if mean itself has NaNs (prev & after both NaN there)
        if np.isnan(mean).any():
            mean[np.isnan(mean)] = np.nanmean(arr[max(day-3, 0):day+1])

        # still might happen if entire slice is NaN
        if np.isnan(elem).all():
            elem = mean
        else:
            elem[missing] = mean[missing]

        arr[day] = elem

    # Final safety pass — replace any remaining NaN with global mean
    global_mean = np.nanmean(arr)
    arr[np.isnan(arr)] = global_mean

    return arr


def flatten_features(array_3d: np.ndarray) -> np.ndarray:
    """Flatten [time, H, W] → [time, H*W]."""
    if array_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {array_3d.ndim}D.")
    return array_3d.reshape(array_3d.shape[0], -1)


def reshape_to_3d(array_2d: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Reshape [time, H*W] → [time, H, W]."""
    t, h, w = original_shape
    return array_2d.reshape(t, h, w)


def process_satellite_data(
    input_path: str,
    output_dir: str,
    train_days: int = 730,
    impute: bool = True,
    scale: bool = True
):
    """Full preprocessing pipeline for satellite raster time series."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load GeoTIFF ---
    with rasterio.open(input_path) as src:
        data = src.read()  # shape [bands=time, H, W]
        meta = src.meta

    print(f"Loaded raster {input_path} → shape {data.shape}")

    # --- Impute missing pixels ---
    if impute:
        print("→ Imputing missing pixels ...")
        data = impute_missing_pixels(data)

    # --- Split train/test ---
    train = data[:train_days]
    test = data[train_days:]

    # --- Flatten for scaling ---
    if scale:
        print("→ Scaling with StandardScaler ...")
        scaler = StandardScaler()
        train_flat = flatten_features(train)
        test_flat = flatten_features(test)

        train_scaled = scaler.fit_transform(train_flat)
        test_scaled = scaler.transform(test_flat)

        # Reshape back
        train_z = reshape_to_3d(train_scaled, train.shape)
        test_z = reshape_to_3d(test_scaled, test.shape)
    else:
        train_z, test_z = train, test

    # --- Save as .npy ---
    base = os.path.splitext(os.path.basename(input_path))[0]
    np.save(os.path.join(output_dir, f"{base}_train.npy"), train_z)
    np.save(os.path.join(output_dir, f"{base}_test.npy"), test_z)
    print(f"Saved preprocessed arrays: {output_dir}")
    

    return {
        "train": train_z,
        "test": test_z,
        "meta": meta
    }


if __name__ == "__main__":
    input_tif = "./data/satellite/MM_NO2_index.tif"
    output_dir = "./processed_data"

    result = process_satellite_data(
        input_path=input_tif,
        output_dir=output_dir,
        train_days=730
    )
