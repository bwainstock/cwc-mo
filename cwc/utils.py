from io import BytesIO
from pathlib import PurePath
from typing import Any
import polars as pl
import numpy as np
from fitparse import FitFile  # type: ignore[import]


def parse_fit_file(
    fit_file_ish: PurePath | str | BytesIO | bytes,
) -> tuple[pl.DataFrame | None, list[dict[str, Any]] | None]:
    """
    Parse a .fit file and extract relevant data into a polars DataFrame.

    Args:
        fit_file_ish: Path, string, BytesIO, or bytes containing .fit file data

    Returns:
        Tuple containing (DataFrame with processed activity data, list of lap messages)
        Returns (None, None) if parsing fails
    """
    fit_file = _load_fit_file(fit_file_ish)
    if fit_file is None:
        return None, None

    data_messages, lap_messages = _extract_messages(fit_file)

    df = pl.DataFrame(data_messages, infer_schema_length=None)
    if df.is_empty():
        print("No record data found in the .fit file")
        return None, None

    df = _process_dataframe(df)

    # Validate required columns
    if not _validate_required_columns(df):
        return None, None

    return df, lap_messages


def _load_fit_file(fit_file_ish: PurePath | str | BytesIO | bytes) -> FitFile | None:
    """Load and parse the .fit file, returning None if there's an error."""
    try:
        fit_file = FitFile(fit_file_ish)
        fit_file.parse()
        return fit_file
    except Exception as e:
        print(f"Error parsing .fit file: {str(e)}")
        return None


def _extract_messages(
    fit_file: FitFile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract data and lap messages from a parsed fit file."""
    data_messages = []
    lap_messages = []

    for message in fit_file.get_messages():
        if message.name == "record":
            data = {field.name: field.value for field in message}
            data_messages.append(data)
        elif message.name == "lap":
            lap_data = {field.name: field.value for field in message}
            lap_messages.append(lap_data)

    return data_messages, lap_messages


def _process_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Process the raw dataframe, converting units and standardizing column names."""
    # Convert speed to v
    if "speed" in df.columns:
        df = df.with_columns(pl.col("speed").alias("v"))
    elif "enhanced_speed" in df.columns:
        df = df.with_columns(pl.col("enhanced_speed").alias("v"))

    # Convert power to watts
    if "power" in df.columns:
        df = df.with_columns(pl.col("power").alias("watts"))

    # Convert altitude to elevation
    if "altitude" in df.columns:
        df = df.with_columns(pl.col("altitude").alias("elevation"))
    elif "enhanced_altitude" in df.columns:
        df = df.with_columns(pl.col("enhanced_altitude").alias("elevation"))

    # Extract GPS coordinates
    if "position_lat" in df.columns and "position_long" in df.columns:
        # Convert semicircles to degrees (Garmin .fit standard)
        df = df.with_columns(
            [
                (pl.col("position_lat") / (2**31 / 180)).alias("latitude"),
                (pl.col("position_long") / (2**31 / 180)).alias("longitude"),
            ]
        )

    # Ensure timestamp is correctly handled
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp"))

    # Add acceleration if not present
    if "a" not in df.columns:
        df = df.with_columns(pl.Series(name="a", values=accel_calc(df["v"], 1)))

    return resample_data(
        df.select("v", "watts", "elevation", "a", "timestamp", "latitude", "longitude"),
    )


def _validate_required_columns(df: pl.DataFrame) -> bool:
    """Validate that all required columns are present in the dataframe."""
    required_columns = {"timestamp", "v", "watts", "elevation"}

    if not required_columns.issubset(df.columns):
        print(f"Missing required columns: {required_columns - set(df.columns)}")
        return False

    return True


def virtual_slope(
    cda: float,
    crr: float,
    df: pl.DataFrame,
    kg: float,
    rho: float,
    vw: float = 0,
    eta: float = 0.98,
) -> np.ndarray:
    """
    Calculate virtual slope.

    Args:
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pl.DataFrame): DataFrame containing 'v', 'watts', and 'a'
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        eta (float): Drivetrain efficiency

    Returns:
        numpy.ndarray: Virtual slope values
    """
    w = df["watts"].to_numpy() * eta
    acc = df["a"].to_numpy()
    vg = df["v"].to_numpy()

    # Initialize result array with zeros (default slope)
    slope = np.zeros_like(vg, dtype=float)

    # Filter out zero velocities first to avoid division by zero
    valid_idx = np.nonzero(vg)[0]

    if len(valid_idx) > 0:
        # Only process entries with valid velocity
        valid_vg = vg[valid_idx]
        valid_w = w[valid_idx]
        valid_acc = acc[valid_idx]

        # Calculate air velocity
        valid_va = valid_vg + vw

        # Calculate slope for valid entries (no division by zero possible)
        valid_slopes = (
            (valid_w / (valid_vg * kg * 9.807))
            - (cda * rho * valid_va**2 / (2 * kg * 9.807))
            - crr
            - valid_acc / 9.807
        )

        # Assign results back to full array
        slope[valid_idx] = valid_slopes

    return slope


def delta_ve(
    cda: float,
    crr: float,
    df: pl.DataFrame,
    kg: float,
    rho: float,
    vw: float = 0,
    dt: float = 1,
    eta: float = 0.98,
) -> np.ndarray:
    """
    Calculate virtual elevation change.

    Args:
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pl.DataFrame): DataFrame containing 'v', 'watts', and 'a'
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        numpy.ndarray: Virtual elevation changes
    """

    # Use our fixed virtual_slope function to calculate slope
    slope = virtual_slope(cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, eta=eta)

    # Calculate virtual elevation change
    return df["v"].to_numpy() * dt * np.sin(np.arctan(slope))


def accel_calc(v, dt):
    """
    Calculate acceleration from velocity data.

    Args:
        v (array-like): Velocity in m/s
        dt (float): Time interval in seconds

    Returns:
        numpy.ndarray: Acceleration values
    """
    v = np.array(v)
    # Replace NaN or zero values with a small positive number
    v[np.isnan(v) | (v < 0.001)] = 0.001

    # Calculate acceleration
    a = np.zeros_like(v)
    for i in range(1, len(v)):
        # Safe division by ensuring denominator is never zero
        if v[i] < 0.001:  # Additional safety check
            a[i] = 0
        else:
            a[i] = (v[i] ** 2 - v[i - 1] ** 2) / (2 * v[i] * dt)

    # Clean up any invalid values
    a[np.isnan(a)] = 0
    a[np.isinf(a)] = 0

    return a


def calculate_virtual_profile(
    ve_changes: np.ndarray, actual_elevation: np.ndarray
) -> np.ndarray:
    """
    Helper function to build the virtual elevation profile from elevation changes.
    This avoids duplicating code in the optimization functions.

    Args:
        ve_changes (numpy.ndarray): Virtual elevation changes
        actual_elevation (numpy.ndarray): Actual elevation data

    Returns:
        numpy.ndarray: Virtual elevation profile
    """
    virtual_profile = np.zeros_like(actual_elevation)

    # For single lap analysis: standard cumulative calculation
    virtual_profile[0] = actual_elevation[0]
    for i in range(1, len(virtual_profile)):
        virtual_profile[i] = virtual_profile[i - 1] + ve_changes[i - 1]

    return virtual_profile


def resample_data(
    df: pl.DataFrame, time_column: str = "timestamp", resample_freq: str = "1s"
):
    """
    Resample data to a constant time interval.

    Args:
        df (pl.DataFrame): DataFrame containing time series data
        resample_freq (str): Resampling frequency (e.g., '1s' for 1 second)

    Returns:
        pandas.DataFrame: Resampled DataFrame
    """

    # Resample to constant time interval
    resampled_df = df.upsample(
        time_column=time_column, every=resample_freq
    ).interpolate()

    # Remove any NaN values
    resampled_df = resampled_df.drop_nans()

    return resampled_df

def calculate_distance(df: pl.DataFrame, dt: float = 1) -> np.ndarray:
    """
    Calculate distance from velocity data.

    Args:
        df (pl.DataFrame): DataFrame containing velocity data
        dt (float): Time interval in seconds

    Returns:
        numpy.ndarray: Distance values
    """
    # Ensure velocity is in m/s
    segment_distances = df["v"].to_numpy() * dt

    return  np.cumsum(segment_distances)