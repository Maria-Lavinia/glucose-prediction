def preprocess_patient(df, patient_id, interp_limit=2):
    # ensure sorted
    df = df.sort_index()

    # resample to 5-min grid
    df = df.resample("5min").mean()

    # interpolate only small gaps
    df["glucose"] = df["glucose"].interpolate(
        method="linear",
        limit=interp_limit
    )

    # add patient id
    df["patient_id"] = patient_id

    return df
