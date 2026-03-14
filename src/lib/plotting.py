import os

def plot_single_loss_curve(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots")
        return

    plot_dir = result.get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(result["train_losses"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, result["train_losses"], label="train")
    plt.plot(epochs, result["val_losses"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {result['name']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(plot_dir, f"loss_{result['name']}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print("Saved plot:", out_path)

def plot_val_comparison(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots")
        return

    if not results:
        return

    plot_dir = results[0].get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for result in results:
        epochs = range(1, len(result["val_losses"]) + 1)
        plt.plot(epochs, result["val_losses"], label=result["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = os.path.join(plot_dir, "val_loss_comparison.png")
    plt.tight_layout()
    plt.savefig(comp_path, dpi=140)
    plt.close()
    print("Saved plot:", comp_path)
