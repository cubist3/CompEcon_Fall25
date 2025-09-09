from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

csv_path = r"C:\Users\Emma Cho\OneDrive - University of South Carolina\Desktop\Data\Data\panel_new.csv"
outdir   = Path(r"C:\Users\Emma Cho\OneDrive - University of South Carolina\Desktop\Data\Data")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

if "month_date" in df.columns:
    month = pd.to_datetime(df["month_date"], errors="coerce")
else:
    year_col  = "YEAROFSALE" if "YEAROFSALE" in df.columns else ("YEAR_OF_SALE" if "YEAR_OF_SALE" in df.columns else None)
    month_col = "MONTHOFSALE" if "MONTHOFSALE" in df.columns else ("MONTH_OF_SALE" if "MONTH_OF_SALE" in df.columns else None)
    if year_col is None or month_col is None:
        raise ValueError("Need 'month_date' OR (YEAROFSALE/YEAR_OF_SALE and MONTHOFSALE/MONTH_OF_SALE).")
    y  = pd.to_numeric(df[year_col], errors="coerce")
    mo = pd.to_numeric(df[month_col], errors="coerce")
    month = pd.to_datetime({"year": y, "month": mo, "day": 1}, errors="coerce")

# dataframe to make desired aggregated figure
# if "State" not in df.columns:
#    raise ValueError("Column 'State' is required.")
df["_month"] = month.dt.to_period("M").dt.to_timestamp()
df = df.dropna(subset=["State", "_month"])

state_month_sales = (
    df.groupby(["State", "_month"], as_index=False)
      .size()
      .rename(columns={"size": "sales"}))

# save the aggregated table for ensure things
(state_month_sales.sort_values(["State", "_month"])
                  .to_csv(outdir / "state_month_sales.csv", index=False))

# pick top 10 states by total sales
TOP_N = 10
top_states = (state_month_sales.groupby("State")["sales"].sum()
    .sort_values(ascending=False).head(TOP_N).index.tolist())

plot_df = state_month_sales[state_month_sales["State"].isin(top_states)]
wide = (plot_df.pivot(index="_month", columns="State", values="sales")
               .sort_index())

# figure 1
fig = plt.figure(figsize=(12.8, 8.0))
ax = fig.add_subplot(111)

for st in wide.columns:
    ax.plot(wide.index, wide[st], linewidth=2, marker="o", markersize=3, label=st)

ax.set_title(f"Monthly Sales by State (Top {len(wide.columns)})", pad=12)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly Sales (count of transactions)")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()

plot1 = outdir / "fig1_state_monthly_sales_lines.png"
plt.savefig(plot1, dpi=150)
plt.show()
print(f"[OK] Saved: {plot1}")

# figure 2
fig = plt.figure(figsize=(13.5, 7.5))
ax = fig.add_subplot(111)

im = ax.imshow(wide.T.values, aspect="auto", interpolation="nearest")
ax.set_title(f"Monthly Sales by State (Heatmap, Top {len(wide.columns)})", pad=12)
ax.set_xlabel("Month")
ax.set_ylabel("State")

# x ticks as months (12 month limits for interpretability / readability)
xticks = np.linspace(0, wide.shape[0] - 1, min(12, max(1, wide.shape[0])), dtype=int)
ax.set_xticks(xticks)
ax.set_xticklabels([wide.index[i].strftime("%Y-%m") for i in xticks], rotation=30, ha="right")

ax.set_yticks(np.arange(wide.shape[1]))
ax.set_yticklabels(wide.columns)

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.ax.set_ylabel("Monthly Sales", rotation=90, labelpad=10)

plt.tight_layout()
plot2 = outdir / "fig2_state_monthly_sales_heatmap.png"
plt.savefig(plot2, dpi=150)
plt.show()
print(f"[OK] Saved: {plot2}")

# figure 3
totals = (state_month_sales.groupby("State", as_index=False)["sales"]
          .sum()
          .sort_values("sales", ascending=False))

fig = plt.figure(figsize=(12.5, 7.5))
ax = fig.add_subplot(111)
ax.bar(totals["State"], totals["sales"])
ax.set_title("Total Sales by State", pad=12)
ax.set_ylabel("Sales (count of transactions)")
ax.set_xlabel("State")
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
plt.tight_layout()

plot3 = outdir / "fig3_state_total_sales_bar.png"
plt.savefig(plot3, dpi=150)
plt.show()
print(f"[OK] Saved: {plot3}")

print("[DONE] Aggregated table ->", outdir / "state_month_sales.csv")
print("[DONE] PNGs ->", plot1, plot2, plot3)

