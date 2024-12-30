import json
import pandas as pd

abs_path = "/home/akson/gesund-sdk-refactor/tests"
with open(f"{abs_path}/_data/classification/test_metadata.json", "r") as f:
    data = json.load(f)


final_results = []
for image_id in data:
    sample = data[image_id]
    metadata = sample["metadata"]
    metadata["image_id"] = image_id
    final_results.append(metadata)

final_results = pd.DataFrame.from_records(final_results)
final_results.to_json(
    f"{abs_path}/_data/classification/test_metadata_new.json", orient="records"
)

print(final_results.head())

reload_df = pd.read_json(f"{abs_path}/_data/classification/test_metadata_new.json")
print(reload_df.head())
