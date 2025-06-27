import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import geopandas as gpd

# Set dataset path
dataset_path = "dataset"

# Initialize containers
label_list = []
inspection_texts = []
geometry_diffs = []
homes = glob.glob(os.path.join(dataset_path, "*"))

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each home
geometries = []
for home in homes:
    try:
        geojson_path = os.path.join(home, "cleaned.geojson")
        results_path = os.path.join(home, "label.json")

        # Load inspection note
        with open(geojson_path) as f:
            geojson = json.load(f)
            inspection_note = geojson["features"][0]["properties"].get("inspection_note", "")
            geometry = shape(geojson["features"][0]["geometry"])
            geometries.append(geometry)
            inspection_texts.append(inspection_note)

        # Load label data
        with open(results_path) as f:
            label_data = json.load(f)
            label_list.append(list(label_data.values()))

    except Exception as e:
        print(f"Skipping {home} due to error: {e}")

# Convert to arrays
labels_array = np.array(label_list)
text_embeddings = model.encode(inspection_texts)
similarity_matrix = cosine_similarity(text_embeddings)
avg_similarity = np.mean(similarity_matrix)

# Geometry differences: compute pairwise Hausdorff distances
hausdorff_distances = []
area_differences = []
for i in range(len(geometries)):
    for j in range(i+1, len(geometries)):
        hausdorff = geometries[i].hausdorff_distance(geometries[j])
        area_diff = abs(geometries[i].area - geometries[j].area)
        hausdorff_distances.append(hausdorff)
        area_differences.append(area_diff)

# Display results
print(f"Number of valid homes (m): {len(label_list)}")
print(f"Average pairwise inspection note similarity: {avg_similarity:.3f}")
print(f"Average Hausdorff distance between homes: {np.mean(hausdorff_distances):.4f}")
print(f"Average area difference between homes: {np.mean(area_differences):.4f}")

print("\nLabel Summary Statistics:")
print(f"Means: {np.mean(labels_array, axis=0)}")
print(f"Standard Deviations: {np.std(labels_array, axis=0)}")
print(f"Mins: {np.min(labels_array, axis=0)}")
print(f"Maxs: {np.max(labels_array, axis=0)}")