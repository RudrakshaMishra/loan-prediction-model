import pandas as pd

# ---------- SETTINGS ----------
INPUT_FILE = "loan.csv"          # Change to your big file name
OUTPUT_FILE = "newloanfile.csv"               # Final output
RANDOM_ROWS = 100000                        # Non-default rows
DEFAULT_ROWS = 40000                        # Charged Off rows
STATUS_COL = "loan_status"                  # Column name

# ---------- LOAD BIG CSV IN CHUNKS ----------
chunks = []
default_chunks = []

print("Reading large file in chunks...")

for chunk in pd.read_csv(INPUT_FILE, chunksize=200000):
    # Charged Off rows
    default_part = chunk[chunk[STATUS_COL] == "Charged Off"]
    if not default_part.empty:
        default_chunks.append(default_part)

    # Random rows (any status)
    chunks.append(chunk)

print("Finished scanning file.")

# ---------- CONCATENATE FOUND DEFAULTS ----------
df_default = pd.concat(default_chunks, ignore_index=True)
print(f"Found total defaults: {len(df_default)}")

# Oversample (with replacement) if not enough defaults
df_default_balanced = df_default.sample(
    n=DEFAULT_ROWS,
    replace=True,
    random_state=42
)

# ---------- SAMPLE RANDOM NON-DEFAULT ROWS ----------
df_all = pd.concat(chunks, ignore_index=True)

df_random = df_all.sample(
    n=RANDOM_ROWS,
    replace=False,
    random_state=42
)

print(f"Random rows taken: {len(df_random)}")
print(f"Oversampled default rows taken: {len(df_default_balanced)}")

# ---------- MERGE BOTH ----------
df_final = pd.concat([df_random, df_default_balanced], ignore_index=True)

print(f"Final dataset size: {len(df_final)}")

# ---------- SAVE ----------
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved reduced file as: {OUTPUT_FILE}")
print("Done!")
