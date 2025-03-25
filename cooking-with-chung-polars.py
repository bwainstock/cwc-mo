# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair-tiles==0.4.0",
#     "altair==5.5.0",
#     "marimo",
#     "shapely==2.0.7",
#     "geopandas==1.0.1",
#     "polars==1.26.0",
#     "fitparse==1.2.0",
#     "numpy==2.2.4",
#     "fitdecode==0.10.0",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from io import BytesIO
    from pathlib import PurePath
    from typing import Any
    import polars as pl
    import pandas as pd
    import numpy as np
    import fitdecode
    return Any, BytesIO, PurePath, fitdecode, mo, np, pd, pl


@app.cell(hide_code=True)
def _(fitdecode, np, pl):
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
                (valid_w / (valid_vg * kg * 9.807)) - (cda * rho * valid_va**2 / (2 * kg * 9.807)) - crr - valid_acc / 9.807
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


    def calculate_virtual_profile(ve_changes: np.ndarray, actual_elevation: np.ndarray) -> np.ndarray:
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


    def resample_data(df: pl.DataFrame, time_column: str = "timestamp", resample_freq: str = "1s"):
        """
        Resample data to a constant time interval.

        Args:
            df (pl.DataFrame): DataFrame containing time series data
            resample_freq (str): Resampling frequency (e.g., '1s' for 1 second)

        Returns:
            pandas.DataFrame: Resampled DataFrame
        """

        # Resample to constant time interval
        resampled_df = df.upsample(time_column=time_column, every=resample_freq).interpolate()

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


    def fit_to_dataframe(fit_file_path):
        """
        Convert a .fit file to a Polars DataFrame, focusing on 'record' message types.

        Args:
            fit_file_path (str): Path to the .fit file

        Returns:
            pl.DataFrame: DataFrame containing record messages
        """
        # List to store record messages
        record_messages = []

        # Open the FIT file
        with fitdecode.FitReader(fit_file_path) as fit_file:
            for frame in fit_file:
                # Skip non-data-message frames or non-record messages
                if not isinstance(frame, fitdecode.FitDataMessage) or frame.name != "record":
                    continue

                # Convert record message to a dictionary
                record_dict = {}

                # Extract all fields from the record message
                for field in frame.fields:
                    record_dict[field.name] = field.value

                record_messages.append(record_dict)

        # Convert to Polars DataFrame
        df = pl.DataFrame(record_messages, infer_schema_length=None)
        df = _process_dataframe(df)

        return df
    return (
        accel_calc,
        calculate_distance,
        calculate_virtual_profile,
        delta_ve,
        fit_to_dataframe,
        resample_data,
        virtual_slope,
    )


@app.cell
def _(mo):
    kg = mo.ui.switch(label="kg")
    return (kg,)


@app.cell
def _(kg, mo):
    weight = mo.ui.number(value=50, label=f"Weight ({'kg' if kg.value else 'lbs'})")
    return (weight,)


@app.cell
def _(mo):
    rho = mo.ui.number(value=1.2, label=f"Air density in kg/m³")
    return (rho,)


@app.cell
def _(mo):
    cda_min, cda_max = 0.1, 0.5
    get_cda, set_cda = mo.state(cda_min)
    return cda_max, cda_min, get_cda, set_cda


@app.cell
def _(cda_max, cda_min, get_cda, mo, set_cda):
    cda_input = mo.ui.number(start=cda_min, stop=cda_max, step=0.001, value=get_cda(), on_change=set_cda)
    return (cda_input,)


@app.cell
def _(cda_max, cda_min, get_cda, mo, set_cda):
    cda_slider = mo.ui.slider(start=cda_min, stop=cda_max, step=0.001, label="CdA", value=get_cda(), on_change=set_cda)
    return (cda_slider,)


@app.cell
def _(mo):
    crr_min, crr_max = 0.001, 0.01
    get_crr, set_crr = mo.state(crr_min)
    return crr_max, crr_min, get_crr, set_crr


@app.cell
def _(crr_max, crr_min, get_crr, mo, set_crr):
    crr_input = mo.ui.number(start=crr_min, stop=crr_max, value=get_crr(), on_change=set_crr)
    return (crr_input,)


@app.cell
def _(crr_max, crr_min, get_crr, mo, set_crr):
    crr_slider = mo.ui.slider(start=crr_min, stop=crr_max, step=0.0005, label="Crr", value=get_crr(), on_change=set_crr)
    return (crr_slider,)


@app.cell
def _(cda_input, cda_slider, crr_input, crr_slider, mo):
    crr = mo.hstack([crr_slider, crr_input])
    cda = mo.hstack([cda_slider, cda_input])
    return cda, crr


@app.cell
def _(mo):
    fit_file = mo.ui.file(filetypes=[".fit"], label="Upload FIT File")
    return (fit_file,)


@app.cell
def _(cda, crr, fit_file, kg, mo, rho, weight):
    unit_picker = mo.hstack([mo.md("lbs"), kg])
    mass_ui = mo.hstack([unit_picker, weight], justify="center")
    mo.vstack([mass_ui, rho, fit_file, cda, crr], align="center").callout("info")
    return mass_ui, unit_picker


@app.cell
def _(accel_calc, calculate_distance, fit_to_dataframe, resample_data):
    raw_df = fit_to_dataframe("/Users/bwainsto/Downloads/good.fit")
    raw_df = raw_df.with_columns(a=accel_calc(raw_df["v"], 1))
    df = resample_data(raw_df)
    distance = calculate_distance(df)
    return df, distance, raw_df


@app.cell
def _(delta_ve, df, get_cda, get_crr, rho, weight):
    ve_changes = delta_ve(cda=get_cda(), crr=get_crr(), df=df, vw=0, kg=weight.value, rho=rho.value, dt=1, eta=0.98)
    return (ve_changes,)


@app.cell
def _(calculate_virtual_profile, df, distance, ve_changes):
    virtual_elevation = calculate_virtual_profile(ve_changes, df["elevation"])
    vdf = df.with_columns(virtual_elevation=virtual_elevation, distance=distance)
    return vdf, virtual_elevation


@app.cell
def _(vdf):
    elevation_df = vdf.unpivot(
        index="distance", on=["elevation", "virtual_elevation"], variable_name="Type", value_name="Value"
    )
    return (elevation_df,)


@app.cell
def _(elevation_df, mo):
    import altair as alt

    brush = alt.selection_interval(encodings=["x"], zoom=True)
    _chart = (
        alt.Chart(elevation_df)
        .mark_line()
        .encode(
            x=alt.X("distance", title="Distance").scale(zero=False),
            y=alt.Y("Value:Q", title="Elevation").scale(zero=False),
            color="Type:N",
        )
        .add_params(brush)
        .properties(title="Elevation vs Virtual Elevation")
    )
    chart = mo.ui.altair_chart(_chart)
    chart
    return alt, brush, chart


@app.cell
def _(chart, vdf):
    selected_df = chart.apply_selection(vdf)
    return (selected_df,)


@app.cell
def _():
    # import geopandas as gpd
    # import altair_tiles as til
    # from shapely import Point, LineString

    # line = LineString([Point(lon, lat) for lon, lat in zip(selected_df["longitude"], selected_df["latitude"])])
    # line_geo = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
    # line_chart = alt.Chart(line_geo).mark_geoshape(filled=False, stroke="green", strokeWidth=1).project( type="mercator")
    # geo_chart_with_tiles = til.add_tiles(line_chart).properties(width=500, height=400)

    # start_point = Point(selected_df.select(["longitude", "latitude"]).row(0))
    # point_geo = gpd.GeoDataFrame(geometry=[start_point], crs="EPSG:4326")
    # point_chart = alt.Chart(point_geo).mark_geoshape(filled=False, stroke="blue").project( type="mercator")
    # point_chart_with_tiles = til.add_tiles(line_chart).properties(width=500, height=400)
    # mo.hstack([chart,point_chart], widths=[4,1])
    return


@app.cell
def _():
    0
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
