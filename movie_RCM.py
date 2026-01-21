import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Global configuration
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10
})

np.random.seed(42)

users = [f"User {i}" for i in range(1, 11)]

movie_items   = [f"Movie {i}" for i in range(1, 11)]
book_items    = [f"Book {i}" for i in range(1, 11)]
music_items   = [f"Track {i}" for i in range(1, 11)]
product_items = [f"Product {i}" for i in range(1, 11)]

# -----------------------------
# Sparse matrix generator
# -----------------------------
def create_sparse_matrix(size=10, density=0.1, max_value=5):
    matrix = np.full((size, size), np.nan)
    num_values = int(size * size * density)
    indices = np.random.choice(size * size, num_values, replace=False)
    for idx in indices:
        r, c = divmod(idx, size)
        matrix[r, c] = np.random.randint(1, max_value + 1)
    return matrix

movie_matrix   = create_sparse_matrix(density=0.12)
book_matrix    = create_sparse_matrix(density=0.07)
music_matrix   = create_sparse_matrix(density=0.15)
product_matrix = create_sparse_matrix(density=0.05)

# -----------------------------
# Colormap (gray = missing, green = interaction)
# -----------------------------
greens = plt.cm.Greens(np.linspace(0.3, 0.9, 5))
cmap = ListedColormap(np.vstack(([0.9, 0.9, 0.9, 1.0], greens)))

# -----------------------------
# Plot function (NO legend text)
# -----------------------------
def plot_matrix(ax, data, title, xlabels, ylabels):
    masked = np.nan_to_num(data, nan=0)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=5)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

# -----------------------------
# Create 2×2 figure
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "Sparse User–Item Interaction Matrices (Gray Cells = Missing Values)",
    fontsize=14,
    y=0.97
)

plot_matrix(
    axes[0, 0],
    movie_matrix,
    "Sparse Movie Recommendation Matrix",
    movie_items,
    users
)

plot_matrix(
    axes[0, 1],
    book_matrix,
    "Sparse Book Recommendation Matrix",
    book_items,
    users
)

plot_matrix(
    axes[1, 0],
    music_matrix,
    "Sparse Music Recommendation Matrix",
    music_items,
    users
)

plot_matrix(
    axes[1, 1],
    product_matrix,
    "Sparse E-commerce Purchases Matrix",
    product_items,
    users
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
