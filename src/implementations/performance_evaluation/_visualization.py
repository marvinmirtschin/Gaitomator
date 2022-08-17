import matplotlib.pyplot as plt
import pandas as pd


def plot_data_frames(original_frame, transformed_frame, row_start=None, row_end=None, column_index=None):
    dfs, titles = create_plots_for_data_frame_collection(original_frame, "OG", row_start, row_end, column_index)
    trans_dfs, trans_titles = create_plots_for_data_frame_collection(transformed_frame, "TR", row_start, row_end)
    for i in range(len(dfs)):
        dfs[i].plot(title=titles[i])
        trans_dfs[i].plot(title=trans_titles[i])
    plt.show()


def create_plots_for_data_frame_collection(data_frame: pd.DataFrame, title, row_start=0, row_end=None, column_index=None):
    dfs = []
    titles = []
    for i, row in data_frame.iterrows():
        if not row_end:
            row_end = len(data_frame)
        if column_index:
            dfs.append(row['data_frames'].iloc[row_start:row_end, column_index])
        else:
            dfs.append(row['data_frames'].iloc[row_start:row_end, ])
        titles.append("{}: {}-{}".format(title, row["label_record"], row["label_user"]))
    return dfs, titles
