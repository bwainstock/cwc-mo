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
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from cwc import parse_fit_file, accel_calc, delta_ve, calculate_virtual_profile
    from cwc.utils import resample_data, calculate_distance
    return (
        accel_calc,
        calculate_distance,
        calculate_virtual_profile,
        delta_ve,
        mo,
        parse_fit_file,
        pl,
        resample_data,
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
def _(mo):
    mo.md(
        r"""
        * cda = 0.399172
        * crr = 0.003679
        * rho = 1.158
        """
    )
    return


@app.cell
def _(cda, crr, fit_file, kg, mo, rho, weight):
    unit_picker = mo.hstack([mo.md("lbs"), kg])
    mass_ui = mo.hstack([unit_picker, weight], justify="center")
    mo.vstack([mass_ui, rho, fit_file, cda, crr], align="center").callout("info")
    return mass_ui, unit_picker


@app.cell
def _(accel_calc, calculate_distance, parse_fit_file, resample_data):
    raw_df, lap_messages = parse_fit_file(("/Users/bwainsto/Downloads/good.fit"))
    raw_df = raw_df.with_columns(a=accel_calc(raw_df["v"], 1))
    df = resample_data(raw_df)
    distance = calculate_distance(df)
    return df, distance, lap_messages, raw_df


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
            x=alt.X("distance", title="Timestamp").scale(zero=False),
            y=alt.Y("Value:Q", title="Elevation").scale(zero=False),
            color="Type:N",
        )
        .add_params(brush)
        .properties(title="Elevation vs Virtual Elevation Over Time")
    )
    chart = mo.ui.altair_chart(_chart)
    return alt, brush, chart


@app.cell
def _(chart, vdf):
    selected_df = chart.apply_selection(vdf)
    return (selected_df,)


@app.cell
def _(alt, chart, mo, selected_df):
    import geopandas as gpd
    import altair_tiles as til
    from shapely import Point, LineString

    line = LineString([Point(lon, lat) for lon, lat in zip(selected_df["longitude"], selected_df["latitude"])])
    start_point = Point(selected_df.select(["longitude", "latitude"]).row(0))
    geo = gpd.GeoDataFrame(geometry=[line, start_point], crs="EPSG:4326")
    geo_chart = alt.Chart(geo).mark_geoshape(fillOpacity=0, stroke="green", strokeWidth=1).project( type="mercator")
    geo_chart_with_tiles = til.add_tiles(geo_chart).properties(width=500, height=400)
    mo.hstack([chart,geo_chart_with_tiles], widths=[4,1])
    return (
        LineString,
        Point,
        geo,
        geo_chart,
        geo_chart_with_tiles,
        gpd,
        line,
        start_point,
        til,
    )


@app.cell
def _():
    0
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
