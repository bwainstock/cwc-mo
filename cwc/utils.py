from io import BytesIO
from pathlib import PurePath
from typing import Any
import polars as pl
import numpy as np
from fitparse import FitFile  # type: ignore[import]
from scipy.optimize import basinhopping, differential_evolution, minimize_scalar
from scipy.stats import pearsonr


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

    return np.cumsum(segment_distances)


def optimize_both_params_balanced(
    df: pl.DataFrame,
    kg: float,
    rho: float,
    n_grid: int = 100,  # Reduced grid size to compensate for multiple starts
    r2_weight: float = 0.5,  # Weight for R² in the composite objective (0-1)
    dt: float = 1,
    eta: float = 0.98,
    vw: float = 0,
    cda_bounds: tuple[float, float] = (
        0.1,
        0.5,
    ),  # CdA typically between 0.1 and 0.5 m²
    crr_bounds: tuple[float, float] = (
        0.001,
        0.01,
    ),  # Crr typically between 0.001 and 0.01
    n_random_starts: int = 5,  # Number of random starting points
    basin_hopping_steps: int = 30,  # Number of basin hopping steps
    basin_hopping_temp: float = 1.0,  # Temperature parameter for basin hopping
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Optimize both CdA and Crr using a balanced approach that combines multiple
    global optimization strategies to avoid local minima.

    Args:
        df (pl.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_grid (int): Number of grid points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, optimized_crr, rmse, r2, virtual_profile)
    """

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(df["elevation"].to_numpy())

    # Define parameter bounds
    bounds = [cda_bounds, crr_bounds]

    # Calculate baseline values and scaling factor for RMSE
    initial_cda = (cda_bounds[0] + cda_bounds[1]) / 2  # Midpoint of cda range
    initial_crr = (crr_bounds[0] + crr_bounds[1]) / 2  # Midpoint of crr range

    initial_ve_changes = delta_ve(
        cda=initial_cda,
        crr=initial_crr,
        df=df,
        vw=vw,
        kg=kg,
        rho=rho,
        dt=dt,
        eta=eta,
    )
    initial_virtual_profile = calculate_virtual_profile(
        initial_ve_changes, actual_elevation
    )
    baseline_rmse = np.sqrt(np.mean((initial_virtual_profile - actual_elevation) ** 2))
    # Use larger of baseline_rmse or 10% of elevation range as scale
    elev_range = np.max(actual_elevation) - np.min(actual_elevation)
    rmse_scale = max(baseline_rmse, 0.1 * elev_range)

    # Define the composite objective function
    def objective(params: np.ndarray) -> float:
        """Objective function that balances R² and RMSE"""
        cda, crr = params

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale  # type: ignore

        # Weighted objective: lower is better
        # Use (1-R²) since we want to maximize R² but minimize the objective
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter set
    def calculate_metrics(cda: float, crr: float) -> tuple[float, float, np.ndarray]:
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    # Create a grid of points to evaluate
    cda_grid = np.linspace(cda_bounds[0], cda_bounds[1], int(np.sqrt(n_grid)))
    crr_grid = np.linspace(crr_bounds[0], crr_bounds[1], int(np.sqrt(n_grid)))

    # Evaluate objective function at each grid point
    grid_results: list[tuple[float, float, float, float, float]] = []
    for cda in cda_grid:
        for crr in crr_grid:
            # Get weighted objective
            weighted_obj = objective([cda, crr])

            # Calculate individual metrics for reporting
            r2, rmse, _ = calculate_metrics(cda, crr)

            grid_results.append((cda, crr, weighted_obj, r2, rmse))

    # Sort grid points by objective (ascending - lower is better)
    grid_results.sort(key=lambda x: x[2])

    # Initialize storage for global best results across all optimization attempts
    global_best_results: list[tuple[float, float, float, float, float, np.ndarray]] = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x: np.ndarray) -> np.ndarray:
        # Use a smaller step size as we get closer to convergence
        step_size = np.array(
            [
                (cda_bounds[1] - cda_bounds[0]) * 0.1,
                (crr_bounds[1] - crr_bounds[0]) * 0.1,
            ]
        )
        # Random step within bounds
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds
        new_x[0] = np.clip(new_x[0], cda_bounds[0], cda_bounds[1])
        new_x[1] = np.clip(new_x[1], crr_bounds[0], crr_bounds[1])
        return new_x

    # Define acceptance test function for basin hopping
    def accept_test(
        f_new: float, x_new: np.ndarray, f_old: float, x_old: np.ndarray
    ) -> bool:
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        # More likely to accept small deteriorations
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    # Start with top grid results and add random starts
    starting_points: list[tuple[float, float]] = [
        (grid_results[i][0], grid_results[i][1])
        for i in range(min(3, len(grid_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append((random_cda, random_crr))

    # Run optimization from each starting point
    for start_idx, (start_cda, start_crr) in enumerate(starting_points):
        # Use differential evolution first for global search
        de_result = differential_evolution(
            objective,
            bounds,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            strategy="best1bin",
            tol=0.01,
            maxiter=100,  # Limit iterations for speed
            init="sobol",  # Use Sobol sequence for better coverage
            updating="deferred",  # Update after a generation
            workers=1,
        )

        # Use basin hopping for exploring multiple basins
        bh_minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"ftol": 1e-6, "gtol": 1e-5},
        }

        bh_result = basinhopping(
            objective,
            de_result.x,  # Start from DE result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, our custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=bh_minimizer_kwargs,
        )

        bh_cda, bh_crr = bh_result.x
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_cda, bh_crr)

        # Store result from this attempt
        global_best_results.append((bh_cda, bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result across all attempts
    global_best_results.sort(key=lambda x: x[2])  # Sort by objective value
    best_cda, best_crr, _, best_r2, best_rmse, best_profile = global_best_results[0]

    return best_cda, best_crr, best_rmse, best_r2, best_profile


def optimize_crr_only_balanced(
    df: pl.DataFrame,
    fixed_cda: float,
    kg: float,
    rho: float,
    n_points: int = 100,  # Reduced for efficiency with multiple starts
    r2_weight: float = 0.5,  # Weight for R² in the composite objective (0-1)
    dt: float = 1,
    eta: float = 0.98,
    vw: float = 0,
    crr_bounds: tuple[float, float] = (
        0.001,
        0.01,
    ),  # Crr typically between 0.001 and 0.01
    n_random_starts: int = 5,  # Number of random starting points
    basin_hopping_steps: int = 30,  # Number of basin hopping steps
    basin_hopping_temp: float = 1.0,  # Temperature parameter for basin hopping
):
    """
    Optimize only Crr with a fixed CdA value using a balanced approach with
    multiple optimization strategies to avoid local minima.

    Args:
        df (pl.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_cda (float): Fixed CdA value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_points (int): Number of points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (fixed_cda, optimized_crr, rmse, r2, virtual_profile)
    """

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(df["elevation"].to_numpy())

    # Calculate baseline values and scaling factor for RMSE
    initial_crr = (crr_bounds[0] + crr_bounds[1]) / 2  # Midpoint of crr range

    initial_ve_changes = delta_ve(
        cda=fixed_cda, crr=initial_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
    )
    initial_virtual_profile = calculate_virtual_profile(
        initial_ve_changes, actual_elevation
    )
    baseline_rmse = np.sqrt(np.mean((initial_virtual_profile - actual_elevation) ** 2))
    # Use larger of baseline_rmse or 10% of elevation range as scale
    elev_range = np.max(actual_elevation) - np.min(actual_elevation)
    rmse_scale = max(baseline_rmse, 0.1 * elev_range)

    # Define the composite objective function
    def objective(crr):
        """Objective function that balances R² and RMSE"""
        # Handle single value input for basin_hopping compatibility
        # Handle array input (ensure we're using a scalar if there's just one element)
        if hasattr(crr, "__len__") and len(crr) == 1:
            crr = float(crr[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale

        # Weighted objective: lower is better
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter
    def calculate_metrics(crr):
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    # Create a linearly spaced array of Crr values to test
    crr_values = np.linspace(crr_bounds[0], crr_bounds[1], n_points)

    # Evaluate the objective function for each Crr value
    initial_results = []
    for crr in crr_values:
        # Get weighted objective
        weighted_obj = objective(crr)

        # Calculate individual metrics for reporting
        r2, rmse, _ = calculate_metrics(crr)

        initial_results.append((crr, weighted_obj, r2, rmse))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (crr_bounds[1] - crr_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, crr_bounds[0], crr_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append(random_crr)

    # Run optimization from each starting point
    for start_idx, start_crr in enumerate(starting_points):
        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(crr_bounds[0], crr_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use basin hopping with starting point as a scalar
        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=crr_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_crr = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_crr)

        # Store result from this attempt
        global_best_results.append((bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_crr, best_obj, best_r2, best_rmse, best_profile = global_best_results[0]

    return fixed_cda, best_crr, best_rmse, best_r2, best_profile


def optimize_cda_only_balanced(
    df: pl.DataFrame,
    fixed_crr: float,
    kg: float,
    rho: float,
    n_points: int = 100,  # Reduced for efficiency with multiple starts
    r2_weight: float = 0.5,  # Weight for R² in the composite objective (0-1)
    dt: float = 1,
    eta: float = 0.98,
    vw: float = 0,
    cda_bounds: tuple[float, float] = (
        0.1,
        0.5,
    ),  # CdA typically between 0.1 and 0.5 m²
    n_random_starts: int = 5,  # Number of random starting points
    basin_hopping_steps: int = 30,  # Number of basin hopping steps
    basin_hopping_temp: float = 1.0,  # Temperature parameter for basin hopping
):
    """
    Optimize only CdA with a fixed Crr value using a balanced approach with
    multiple optimization strategies to avoid local minima.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_crr (float): Fixed Crr value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_points (int): Number of points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, fixed_crr, rmse, r2, virtual_profile)
    """

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(df["elevation"].to_numpy())

    # Calculate baseline values and scaling factor for RMSE
    initial_cda = (cda_bounds[0] + cda_bounds[1]) / 2  # Midpoint of cda range

    initial_ve_changes = delta_ve(
        cda=initial_cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
    )
    initial_virtual_profile = calculate_virtual_profile(
        initial_ve_changes, actual_elevation
    )
    baseline_rmse = np.sqrt(np.mean((initial_virtual_profile - actual_elevation) ** 2))
    # Use larger of baseline_rmse or 10% of elevation range as scale
    elev_range = np.max(actual_elevation) - np.min(actual_elevation)
    rmse_scale = max(baseline_rmse, 0.1 * elev_range)

    # Define the composite objective function
    def objective(cda):
        """Objective function that balances R² and RMSE"""
        # Handle single value input for basin_hopping compatibility
        # Handle array input (ensure we're using a scalar if there's just one element)
        if hasattr(cda, "__len__") and len(cda) == 1:
            cda = float(cda[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale

        # Weighted objective: lower is better
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter
    def calculate_metrics(cda):
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(ve_changes, actual_elevation)
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    # Create a linearly spaced array of CdA values to test
    cda_values = np.linspace(cda_bounds[0], cda_bounds[1], n_points)

    # Evaluate the objective function for each CdA value
    initial_results = []
    for cda in cda_values:
        # Get weighted objective
        weighted_obj = objective(cda)

        # Calculate individual metrics for reporting
        r2, rmse, _ = calculate_metrics(cda)

        initial_results.append((cda, weighted_obj, r2, rmse))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (cda_bounds[1] - cda_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, cda_bounds[0], cda_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        starting_points.append(random_cda)

    # Run optimization from each starting point
    for start_idx, start_cda in enumerate(starting_points):

        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(cda_bounds[0], cda_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use basin hopping with starting point as a scalar
        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=cda_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_cda = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_cda)

        # Store result from this attempt
        global_best_results.append((bh_cda, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_cda, best_obj, best_r2, best_rmse, best_profile = global_best_results[0]

    return best_cda, fixed_crr, best_rmse, best_r2, best_profile
