# Official protocol evidence

Source: Hongn et al., Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1, PhysioNet (2025), https://physionet.org/content/wearable-device-dataset/1.0.1/. Notebook: https://physionet.org/files/wearable-device-dataset/1.0.1/Wearable_Dataset.ipynb. Dataset file license: Open Data Commons Attribution License v1.0, as documented on the dataset page.

Images are unmodified embedded attachments from the official notebook (zero-based cells 9, 11, 14 and 16). official_mapping_code.txt is the original graph_multiple cell 25, which explicitly identifies TMCT intervals [5,6] for version 1 and [2,3] for version 2. The notebook inserts the recording start at index 0 before the button tags.

The figures support the selected primary rest/stress and active aerobic mappings. Numeric alignment to each participant's tags is checked by recovery.py. This does not independently verify that participants experienced psychological stress or eliminate event-timing errors in the original records.

Important limitation: the version-1 figure shows baseline between tags 1 and 2, whereas the historical selected manifest uses 0 to 1. The earlier automatic Stroop label for interval 1 to 2 is also unsupported by the official code, which uses 3 to 4. Neither interval is used in the primary rest/stress model; the recovery omits baseline-inclusive sensitivity endpoints and does not silently relabel them. Do not claim the old baseline sensitivity is validated.
