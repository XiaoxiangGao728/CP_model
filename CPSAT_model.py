import pandas as pd
from ortools.sat.python import cp_model
import math

# 1. Read the original data
df = pd.read_csv('HitMind反应汇总_FINAL.csv')
# Drop missing values and reset index
df = df[['Sub', 'Prod']].dropna().reset_index(drop=True)

# 2. Construct conflict pairs: if the Sub of one row is in the Prod of another, or vice versa, they cannot be in the same group
n = len(df)
conflicts = []
sub_list = df['Sub'].astype(str).tolist()
prod_list = df['Prod'].astype(str).tolist()
for i in range(n):
    for j in range(i+1, n):
        if (sub_list[i] == prod_list[j]) or (sub_list[j] == prod_list[i]):
            conflicts.append((i, j))

# 3. Calculate the number of target groups k, such that n/k is roughly in [40, 50]
#    Minimum k so that n/k <= 50, maximum k so that n/k >= 40
k_min = math.ceil(n / 50)
k_max = math.floor(n / 40)
if k_min > k_max:
    # If the [40,50] range is infeasible, prioritize the maximum lower bound
    k = k_min
else:
    # Take the middle value
    k = (k_min + k_max) // 2

print(f"Total entries = {n}, attempting to divide into {k} groups (about {n/k:.1f} entries per group)")

# 4. Build the model
model = cp_model.CpModel()

# x[i,g] = 1 indicates row i is assigned to group g
x = {}
for i in range(n):
    for g in range(k):
        x[(i,g)] = model.NewBoolVar(f"x_{i}_{g}")

# Each row is assigned to exactly one group
for i in range(n):
    model.Add(sum(x[(i,g)] for g in range(k)) == 1)

# Group size constraints: 40 <= size <= 50
for g in range(k):
    model.Add(sum(x[(i,g)] for i in range(n)) >= 40)
    model.Add(sum(x[(i,g)] for i in range(n)) <= 50)

# Conflict constraints: conflict pairs (i,j) cannot be in the same group
for i, j in conflicts:
    for g in range(k):
        model.Add(x[(i,g)] + x[(j,g)] <= 1)

# 5. Solve and export
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0  # Max 60 seconds
solver.parameters.num_search_workers = 8

res = solver.Solve(model)
if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
    raise RuntimeError("No feasible solution found within time limit. Please adjust parameters or relax constraints.")

# Assemble results into a wide-format table
grouped = {g: [] for g in range(k)}
for i in range(n):
    for g in range(k):
        if solver.Value(x[(i,g)]) == 1:
            grouped[g].append((df.at[i,'Sub'], df.at[i,'Prod']))
            break

# Build DataFrame and concatenate by columns
all_dfs = []
for g in range(k):
    subs, prods = zip(*grouped[g])
    tmp = pd.DataFrame({
        f"Sub_Group_{g+1}": subs,
        f"Prod_Group_{g+1}": prods
    })
    all_dfs.append(tmp)

out = pd.concat(all_dfs, axis=1)
out.to_csv('All_Groups_Columns_optimal.csv', index=False)
print("Optimal grouping results exported to: All_Groups_Columns_optimal.csv")
