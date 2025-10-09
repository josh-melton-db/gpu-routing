# Databricks notebook source
# DBTITLE 1,Install CuOpt
# MAGIC %pip install -q --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12 cuopt-sh-client cuopt-cu12==25.8.*
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Set Configs
NUM_SHIPMENTS = 20_000
NUM_ROUTES = int(round(NUM_SHIPMENTS / 250 )) # total trucks available
MAX_EV = 4000 # max capacity
MAX_VAN = 8000
DEPOT_LAT, DEPOT_LON = 39.7685, -86.1580 # start and end point for each route
SOLVER_MINUTES = 10


catalog = "default"
schema = f"routing"
shipments_table = f"{catalog}.{schema}.raw_shipments_{NUM_SHIPMENTS}"
mapping_table = f"{catalog}.{schema}.shipment_ids_map_{NUM_SHIPMENTS}"
clustered_table = f"{catalog}.{schema}.shipment_clusters_gpu_{NUM_SHIPMENTS}"
distances_table = f"{catalog}.{schema}.distances_by_route_gpu_{NUM_SHIPMENTS}"
routing_table = f"{catalog}.{schema}.routing_unified_by_cluster_gpu_{NUM_SHIPMENTS}"

# COMMAND ----------

# DBTITLE 1,Check GPU
# MAGIC %sh nvidia-smi

# COMMAND ----------

# DBTITLE 1,Import Libraries
import cudf
from cuopt import routing, distance_engine
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Simple Example
cost = cudf.DataFrame([[0,3,1,2],[3,0,1,2],[2,3,0,2],[2,3,1,0]], dtype='float32')
n_locations = cost.shape[0]
n_vehicles = 2
n_orders = 3  # one order per task node

dm = routing.DataModel(n_locations, n_vehicles, n_orders)
dm.add_cost_matrix(cost)
dm.add_transit_time_matrix(cost.copy(deep=True))  # separate if times differ

ss = routing.SolverSettings()
ss.set_verbose_mode(True)
# ss.set_time_limit(5)
sol = routing.Solve(dm, ss)

print(sol.get_route())      # pandas-like table
sol.display_routes()        # pretty print 

# COMMAND ----------

# DBTITLE 1,Simple Waypoint Example
# Hello World: cuOpt with CSR Waypoint Graph (no dense N×N)

import numpy as np
import cudf
from cuopt import distance_engine, routing


base = np.array([
    [0,3,1,2],
    [3,0,1,2],
    [2,3,0,2],
    [2,3,1,0]
], dtype=np.float32)

V = base.shape[0]
# Build CSR: for each src i, edges to all j != i
indices = []
weights = []
offsets = [0]
for i in range(V):
    for j in range(V):
        if i == j:
            continue
        indices.append(j)
        weights.append(float(base[i, j]))
    offsets.append(len(indices))

indices = np.asarray(indices, dtype=np.int32)        # size E = 12
weights = np.asarray(weights, dtype=np.float32)      # size E
offsets = np.asarray(offsets, dtype=np.int32)        # size V+1

# ----- 2) Build Waypoint graph and compute compact matrix for targets -----
wg = distance_engine.WaypointMatrix(offsets, indices, weights)
targets = np.arange(V, dtype=np.int32)               # use all 4 nodes; 0 will be the depot
cost = wg.compute_cost_matrix(targets)               # cudf.DataFrame (4x4)

# ----- 3) Build a minimal routing model and solve -----
n_locations = len(targets)
n_vehicles  = 2
n_orders    = 3

dm = routing.DataModel(n_locations, n_vehicles, n_orders)

# vehicles start/end at depot (index 0 in our target set)
dm.set_vehicle_locations(
    cudf.Series([0]*n_vehicles),   # starts
    cudf.Series([0]*n_vehicles)    # ends
)

# 3 orders at locations 1,2,3  (indices within the compact matrix)
dm.set_order_locations(cudf.Series([1,2,3]))

# Primary matrices
dm.add_cost_matrix(cost)

# Optional: require both vehicles to be used (just to see multiple routes)
dm.set_min_vehicles(n_vehicles)

ss = routing.SolverSettings()
ss.set_verbose_mode(True)
# ss.set_time_limit(5)  # optional

sol = routing.Solve(dm, ss)
print(sol.get_route())   # pandas-like table
sol.display_routes()     # pretty print

# COMMAND ----------

# DBTITLE 1,Prepare Data
DEPOT_ID = 0

distances_df = (
  spark.read.table(distances_table)
  .select("global_idx_source", "global_idx_dest", "duration_seconds")
)

rev_to_depot = (
    distances_df
      .where(F.col("global_idx_dest") == DEPOT_ID)
      .select(
          F.lit(DEPOT_ID).alias("global_idx_source"),
          F.col("global_idx_source").alias("global_idx_dest"),
          F.col("duration_seconds")
      )
)

distances_df = (
    distances_df
    .unionByName(rev_to_depot)
    # .orderBy("global_idx_source", "global_idx_dest")
)  
display(distances_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Questions**
# MAGIC - Works for 20k packages on a 1 min solver time limit, but do longer solves take more memory?
# MAGIC - How do we set a max duration per truck?
# MAGIC
# MAGIC **Notes**
# MAGIC - OOMs at 40k packages
# MAGIC - TODO: runs trials on CPU vs GPU

# COMMAND ----------

# DBTITLE 1,Solve!
# ---------------------------
# 1) Pull edges and normalize
# ---------------------------
pdf = (
    distances_df
      .select("global_idx_source","global_idx_dest","duration_seconds")
      .toPandas()
)

# Build stable 0..n-1 index space for ALL nodes seen in src or dest
all_nodes = pd.Index(pd.unique(pd.concat([pdf["global_idx_source"],
                                          pdf["global_idx_dest"]], ignore_index=True)))
node2pos = {int(g): i for i, g in enumerate(all_nodes)}
n = len(all_nodes)

pdf["src_idx"] = pdf["global_idx_source"].map(node2pos).astype(np.int32)
pdf["dst_idx"] = pdf["global_idx_dest"].map(node2pos).astype(np.int32)
pdf["cost"]    = pdf["duration_seconds"].astype(np.float32)

# ---------------------------
# 2) Build CSR (offsets/indices/weights)
# ---------------------------
pdf = pdf.sort_values(["src_idx","dst_idx"], kind="mergesort")

indices = pdf["dst_idx"].to_numpy(dtype=np.int32)       # E-length array of neighbor dsts
weights = pdf["cost"].to_numpy(dtype=np.float32)        # E-length array of edge costs

# counts per src over the FULL 0..n-1 range (nodes with 0 out-edges still get an offset)
counts = (
    pdf.groupby("src_idx").size()
       .reindex(range(n), fill_value=0)
       .to_numpy(dtype=np.int32)
)

# offsets[v]..offsets[v+1]-1 slice into `indices`/`weights`
offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int32)

# ---------------------------
# 3) Waypoint graph + compact matrix for selected targets
# ---------------------------
wg = distance_engine.WaypointMatrix(offsets, indices, weights)
order_globals = [int(x) for x in all_nodes if int(x) != DEPOT_ID]

# Targets are the nodes we want in the compact matrix: [depot] + orders
targets = np.array(
    [node2pos[DEPOT_ID]] + [node2pos[g] for g in order_globals],
    dtype=np.int32
)

cost = wg.compute_cost_matrix(targets)   # cudf.DataFrame (len(targets) x len(targets))
time = cost.copy(deep=True)              # use cost as time for now

# ---------------------------
# 4) Routing model and solve
# ---------------------------
n_locations = len(targets)
n_orders    = len(order_globals)

dm = routing.DataModel(n_locations, NUM_ROUTES, n_orders)
dm.set_vehicle_locations(cudf.Series([0]*NUM_ROUTES), cudf.Series([0]*NUM_ROUTES))
dm.set_order_locations(cudf.Series(np.arange(1, n_locations, dtype=np.int32)))

# Primary matrices
dm.add_cost_matrix(cost)
dm.add_transit_time_matrix(time)
dm.set_min_vehicles(NUM_ROUTES)

ss = routing.SolverSettings()
ss.set_verbose_mode(True)
ss.set_time_limit(SOLVER_MINUTES * 60)  # time limit in seconds

sol = routing.Solve(dm, ss)
sol.display_routes()

# COMMAND ----------

# DBTITLE 1,Save Results
route_pdf = sol.get_route().to_pandas()
optimized_routes_df = spark.createDataFrame(route_pdf)
optimized_routes_df.write.mode("overwrite").saveAsTable(routing_table)
display(spark.read.table(routing_table))

# COMMAND ----------

# assert targets[0] == node2pos[DEPOT_ID]       # depot is first
# assert n_orders == (len(targets) - 1)         # one order per non-depot target
# cm = cost.to_pandas().to_numpy()
# unreachable = ~np.isfinite(cm)
# print("Unreachable pairs:", np.argwhere(unreachable))

# COMMAND ----------

# order_global_ids = [int(g) for g in all_nodes if int(g) != DEPOT_ID]

# # (B) Or: if you have a specific set of task nodes (global ids)
# # order_global_ids = my_tasks_global_ids   # e.g., from a Spark/Pandas table

# # Build compact target list: depot first, then exactly the order nodes you want
# targets = np.array(
#     [node2pos[DEPOT_ID]] + [node2pos[g] for g in order_global_ids],
#     dtype=np.int32
# )

# # Compute the compact cost/time matrices over just these targets
# cost = wg.compute_cost_matrix(targets)    # cudf.DataFrame
# time = cost.copy(deep=True)               # or a different matrix if you have it

# # -----------------------------
# # Map orders to compact indices
# # -----------------------------
# # The compact matrix rows/cols are 0..len(targets)-1 with depot at 0
# # If there is exactly ONE order per target (typical VRP), orders = [1..n_locations-1]
# n_locations = len(targets)
# order_locs_compact = np.arange(1, n_locations, dtype=np.int32)

# # (C) Multiple orders at the same physical location?
# # Suppose you have a table with counts per location; repeat that compact index.
# # Example: {global_id: count}
# # counts_by_global = {4150: 2, 2766: 1, 958: 3, ...}
# # order_locs_compact = []
# # for g, cnt in counts_by_global.items():
# #     compact_idx = 1 + order_global_ids.index(g)   # because depot is at 0
# #     order_locs_compact.extend([compact_idx] * cnt)
# # order_locs_compact = np.array(order_locs_compact, dtype=np.int32)

# # -----------------------------
# # Build and solve the model
# # -----------------------------
# n_orders   = len(order_locs_compact)
# n_vehicles = NUM_ROUTES

# dm = routing.DataModel(n_locations, n_vehicles, n_orders)

# # Vehicles start/end at depot (compact index 0)
# dm.set_vehicle_locations(
#     cudf.Series([0]*n_vehicles), 
#     cudf.Series([0]*n_vehicles)
# )

# # Tell cuOpt where each order lives (indices into the compact matrix)
# dm.set_order_locations(cudf.Series(order_locs_compact))

# # Add matrices
# dm.add_cost_matrix(cost)
# dm.add_transit_time_matrix(time)

# # Optional knobs:
# # dm.set_min_vehicles(n_vehicles)          # force using all vehicles
# # dm.set_vehicle_max_time(cudf.Series([...]))  # cap per-vehicle time to spread work
# # dm.set_vehicle_fixed_cost(cudf.Series([...]))# make extra vehicles “cost” something

# ss = routing.SolverSettings()
# ss.set_verbose_mode(True)
# ss.set_time_limit(60)

# sol = routing.Solve(dm, ss)
# sol.display_routes()

# COMMAND ----------

# # neighbors_df: columns [src_idx:int, dst_idx:int, cost:float32], ~50 per src
# # 1) CSR (offsets/indices/weights) of size V=n, E=edges
# n = n_locations
# edges = distances_pdf.sort_values(["global_idx_source","global_idx_dest"])
# indices = edges["global_idx_source"].to_numpy(dtype=np.int32)
# weights = edges["duration_seconds"].to_numpy(dtype=np.float32)

# # offsets: length n+1; offsets[v]..offsets[v+1]-1 slice into `indices`
# counts = edges.groupby("global_idx_source").size().reindex(range(n), fill_value=0).to_numpy(np.int32)
# offsets = np.concatenate([[0], np.cumsum(counts)])

# all_nodes = pd.concat([distances_pdf["global_idx_source"], distances_pdf["global_idx_dest"]]).unique()
# cluster_node_indices = sorted(all_nodes) 

# # 2) Build WaypointMatrix and compute a dense cost matrix only for a subset
# wmat = distance_engine.WaypointMatrix(offsets, indices, weights)
# targets = np.array(cluster_node_indices, dtype=np.int32)          # e.g., a 1–5k node cluster + depot
# cost_mat = wmat.compute_cost_matrix(targets)                      # returns cudf.DataFrame (float32)

# dm = routing.DataModel(cost_mat.shape[0], NUM_ROUTES, max(cost_mat.shape[0]-1, 0))
# dm.add_cost_matrix(cost_mat)  # or dm.add_cost_matrix(cost_mat) depending on version

# # 3) Solve and save results
# ss = routing.SolverSettings()
# ss.set_time_limit(60*20)  # time limit in seconds
# sol = routing.Solve(dm, ss)

# COMMAND ----------

# route_pdf = sol.get_route().to_pandas()
# optimized_routes_df = spark.createDataFrame(route_pdf)
# optimized_routes_df.write.saveAsTable(routing_table)

# COMMAND ----------

# # distances_pdf has columns: [global_idx_source, global_idx_dest, duration_seconds]
# # Build a stable global index space 0..n-1
# all_nodes = pd.Index(
#     pd.unique(
#         pd.concat([distances_pdf["global_idx_source"], 
#                    distances_pdf["global_idx_dest"]], 
#                   ignore_index=True)
#     )
# )
# node2pos = {int(g): i for i, g in enumerate(all_nodes)}
# n = len(all_nodes)

# # Map to 0-based indices for CSR
# edges = distances_pdf[["global_idx_source","global_idx_dest","duration_seconds"]].copy()
# edges["src_idx"] = edges["global_idx_source"].map(node2pos).astype(np.int32)
# edges["dst_idx"] = edges["global_idx_dest"].map(node2pos).astype(np.int32)
# edges["cost"]    = edges["duration_seconds"].astype(np.float32)

# # Sort by source, then destination (helps CSR construction)
# edges = edges.sort_values(["src_idx","dst_idx"], kind="mergesort")

# # CSR pieces
# indices = edges["dst_idx"].to_numpy(dtype=np.int32)        # DESTINATIONS go here
# weights = edges["cost"].to_numpy(dtype=np.float32)         # Edge weights

# # counts per source (include sources with zero neighbors)
# counts = (
#     edges.groupby("src_idx")
#          .size()
#          .reindex(range(n), fill_value=0)
#          .to_numpy(dtype=np.int32)
# )
# offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)  # length n+1

# # Build waypoint graph
# wmat = routing.WaypointMatrix(offsets, indices, weights)

# # Choose targets (must be in same 0..n-1 space). Example: cluster_node_indices are your global_idx values.
# # Map them to positions; also put depot first if you have a known depot global_idx
# targets_pos = np.array([node2pos[int(g)] for g in cluster_node_indices], dtype=np.int32)

# # Compute dense cost matrix ONLY for this subset
# cost_mat = wmat.compute_cost_matrix(targets_pos)  # cudf.DataFrame (float32), shape k x k

# # Solve on the subset
# k = cost_mat.shape[0]
# dm = routing.DataModel(k, NUM_ROUTES, max(k-1, 0))
# # Depending on version: set_matrix or add_cost_matrix
# try:
#     dm.set_matrix(cost_mat)
# except AttributeError:
#     dm.add_cost_matrix(cost_mat)

# ss = routing.SolverSettings()
# ss.set_time_limit(60*20)
# sol = routing.Solve(dm, ss)

# COMMAND ----------

# # === Build a cuOpt-ready cost matrix from distances_df (cost = duration_seconds) ===
# # 1) Get the node list (deterministic order)
# nodes = (
#     distances_df
#       .select(F.col("global_idx_source").alias("id"))
#       .unionByName(distances_df.select(F.col("global_idx_dest").alias("id")))
#       .distinct()
#       .orderBy("id")
#       .toPandas()["id"]
#       .tolist()
# )

# # If you have a known depot global index, put it first:
# nodes.sort(key=lambda x: (x != 0, x))

# n = len(nodes)
# idx_pos = {g:i for i,g in enumerate(nodes)}

# # 2) Initialize matrix with a big finite penalty; 0 on diagonal
# M = np.full((n, n), 1e9, dtype=np.float32)
# np.fill_diagonal(M, 0.0)

# # 3) Fill matrix from the distances table
# pdf = (
#     distances_df
#       .select("global_idx_source", "global_idx_dest", F.col("duration_seconds").alias("cost"))
#       .toPandas()
# )

# for s, d, c in pdf.itertuples(index=False):
#     i = idx_pos[s]; j = idx_pos[d]
#     M[i, j] = np.float32(c)

# cost_gdf = cudf.DataFrame(M)

# # 4) cuOpt model: cost = distance, reuse as transit time (simple case)
# n_locations = n
# n_orders = max(n_locations - 1, 0)  # tasks = all non-depot nodes if you use a depot-first convention

# dm = routing.DataModel(n_locations, NUM_ROUTES, n_orders)
# dm.add_cost_matrix(cost_gdf)
# # dm.add_transit_time_matrix(cost_gdf)

# ss = routing.SolverSettings()
# ss.set_time_limit(60*20)  # time limit in seconds
# sol = routing.Solve(dm, ss)

# # 5) (Optional) map cuOpt node_index -> your global_idx
# route_pdf = sol.get_route().to_pandas()
# route_pdf["global_idx"] = route_pdf["node_index"].map(lambda i: nodes[int(i)])

# COMMAND ----------

# optimized_routes_df = spark.createDataFrame(route_pdf)
# optimized_routes_df.write.saveAsTable(routing_table)
# spark.read.table(routing_table).display()

# COMMAND ----------

# # Map back to your global_idx for readability
# route_pdf = sol.get_route().to_pandas()
# pos2node = {v:k for k,v in node2pos.items()}
# subset_pos2global = {i: int(cluster_node_indices[i]) for i in range(k)}  # target order
# route_pdf["global_idx"] = route_pdf["node_index"].map(lambda i: subset_pos2global[int(i)])

# optimized_routes_df = spark.createDataFrame(route_pdf)
# optimized_routes_df.write.saveAsTable(routing_table)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                | description                                                                                      | license      | source                                                    |
# MAGIC |------------------------|--------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------------|
# MAGIC | OSRM Backend Server    | High performance routing engine written in C++14 designed to run on OpenStreetMap data           | BSD 2-Clause "Simplified" License | https://github.com/Project-OSRM/osrm-backend              |
# MAGIC | osmnx                  | Download, model, analyze, and visualize street networks and other geospatial features from OpenStreetMap in Python | MIT License  | https://github.com/gboeing/osmnx                          |
# MAGIC | ortools                | Operations research tools developed at Google for combinatorial optimization                     | Apache License 2.0 | https://github.com/google/or-tools                        |
# MAGIC | folium                 | Visualize data in Python on interactive Leaflet.js maps                                          | MIT License  | https://github.com/python-visualization/folium            |
# MAGIC | dash                   | Python framework for building analytical web applications and dashboards; built on Flask, React, and Plotly.js | MIT License  | https://github.com/plotly/dash                            |
# MAGIC | branca                 | Library for generating complex HTML+JS pages in Python; provides non-map-specific features for folium | MIT License  | https://github.com/python-visualization/branca            |
# MAGIC | plotly                 | Open-source Python library for creating interactive, publication-quality charts and graphs        | MIT License  | https://github.com/plotly/plotly.py                       |
# MAGIC ray |	Flexible, high-performance distributed execution framework for scaling Python workflows |	Apache2.0 |	https://github.com/ray-project/ray