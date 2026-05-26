import pandas as pd
import random

# ---------- CONFIG ----------
INPUT_CSV = "frames2.csv"
OUTPUT_CSV = "selected_serves2.csv"

SERVES_PER_PLAYER = 12

# Distribution per player 
DISTRIBUTION = {
    "valid": 5,
    "too_high": 3,
    "foot_on_line": 2,
    "foot_lifted": 2
}

RANDOM_SEED = 42
# ----------------------------

random.seed(RANDOM_SEED)

df = pd.read_csv(INPUT_CSV)

# Extract player ID from video name 
df["player"] = df["video"].apply(lambda x: x.split("_")[0])

selected_rows = []

players = df["player"].unique()

for player in players:
    player_df = df[df["player"] == player]

    print(f"\nSelecting serves for {player}")

    player_selected = []

    for label, count in DISTRIBUTION.items():
        label_df = player_df[player_df["label"] == label]

        if len(label_df) < count:
            print(f"Not enough '{label}' serves for {player}. Taking all available.")
            sampled = label_df
        else:
            sampled = label_df.sample(n=count, random_state=RANDOM_SEED)

        player_selected.append(sampled)

    player_selected_df = pd.concat(player_selected)

    # Safety check
    if len(player_selected_df) != SERVES_PER_PLAYER:
        print(f" {player} has {len(player_selected_df)} selected serves instead of {SERVES_PER_PLAYER}")

    selected_rows.append(player_selected_df)

# Combine all players
final_df = pd.concat(selected_rows)

# Save
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(final_df)} selected serves to {OUTPUT_CSV}")
