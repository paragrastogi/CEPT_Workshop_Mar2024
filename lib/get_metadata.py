
import pandas as pd
from lib.reduce_mem_usage import reduce_mem_usage


def get_metadata(path_metadata):

    # Read the buildings metadata file.
    metadata = pd.read_csv(path_metadata + "metadata.csv", usecols = ["building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage", "sqm"])
    # This doesn't change the maths, just makes the dataframe more manageable.
    metadata = reduce_mem_usage(metadata)
    metadata.dropna(how='any',inplace=True)

    return metadata