import json
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 6
})

def filter_data(task, threshold=50002):
    filtered_x = [x for x in task["x"] if x <= threshold]
    filtered_y = [task["y"][i] for i, x in enumerate(task["x"]) if x <= threshold]
    return {"name": task["name"], "x": filtered_x, "y": filtered_y, "type": task["type"], "task": task["task"]}

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def plot_data(ax, task_data, linestyle='-o'):
    for task_name, task_info in task_data.items():
        if "RL" in task_name:
            linestyle='--o'
        ax.plot(task_info["x"], task_info["y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.5',markersize=2)

# Load data from the files
file_paths = [
    'val_imagine1_Voxel_Empty_SemIoU _ val_imagine1_Voxel_Empty_SemIoU.json',
    'val_imagine1_Voxel_IoU _ val_imagine1_Voxel_IoU.json',
    'val_imagine1_Voxel_Precision _ val_imagine1_Voxel_Precision.json',
    'val_imagine1_Voxel_Recall _ val_imagine1_Voxel_Recall.json',
    'val_imagine2_Voxel_Empty_SemIoU _ val_imagine2_Voxel_Empty_SemIoU.json',
    'val_imagine2_Voxel_IoU _ val_imagine2_Voxel_IoU.json',
    'val_imagine2_Voxel_Precision _ val_imagine2_Voxel_Precision.json',
    'val_imagine2_Voxel_Recall _ val_imagine2_Voxel_Recall.json'
]

data_1 = [filter_data(task) for task in load_data(file_paths[0])]
data_2 = [filter_data(task) for task in load_data(file_paths[1])]
data_3 = [filter_data(task) for task in load_data(file_paths[2])]
data_4 = [filter_data(task) for task in load_data(file_paths[3])]
data_5 = [filter_data(task) for task in load_data(file_paths[4])]
data_6 = [filter_data(task) for task in load_data(file_paths[5])]
data_7 = [filter_data(task) for task in load_data(file_paths[6])]
data_8 = [filter_data(task) for task in load_data(file_paths[7])]

# Create a dictionary to store data for each task
task_data_1 = {}
task_data_2 = {}
task_data_3 = {}
task_data_4 = {}
task_data_5 = {}
task_data_6 = {}
task_data_7 = {}
task_data_8 = {}

colors = ['#2E86C1', '#5DADE2', '#D68910', '#F39C12']


# Define a list of colors to use for each task
colors = ['#', '#', '#', '#','#DDCC77','#CC6677','#','#']



colors = ['#117733', '#AA4499', '#332288']
colors2 = ['#44AA99', '#882255', '#88CCEE']

# Process filtered data
for i, (filtered_data, task_data) in enumerate(zip([data_1, data_2, data_3, data_4,data_5, data_6, data_7, data_8], [task_data_1, task_data_2, task_data_3, task_data_4,task_data_5, task_data_6, task_data_7, task_data_8])):
    for j, item in enumerate(filtered_data):
        print(i,j)
        task_name = item["task"]
        if i <= 3:
            color = colors[j % len(colors)]
        else:
            color = colors2[j % len(colors2)]
        if task_name not in task_data:
            task_data[task_name] = {"x": [], "y": [], "name": item["name"], "color": color}

        task_data[task_name]["x"].extend(item["x"])
        task_data[task_name]["y"].extend(item["y"])

# Plot the data for each task in subplots
subplot_width = 1.71875  # 6.875 inches divided by 4 subplots

fig, axs = plt.subplots(1, 4, figsize=(6.875, subplot_width), gridspec_kw={'width_ratios': [subplot_width] * 4})
#fig.set_size_inches(w=6.875, h=1.5)

plot_data(axs[1], task_data_1)
plot_data(axs[0], task_data_2)
plot_data(axs[2], task_data_3)
plot_data(axs[3], task_data_4)
plot_data(axs[1], task_data_5)
plot_data(axs[0], task_data_6)
plot_data(axs[2], task_data_7)
plot_data(axs[3], task_data_8)

# Set titles and labels
axs[0].set_title(f"IoU$^+$ $\\uparrow$")
axs[1].set_title(f"IoU$^-$ $\\uparrow$")
axs[2].set_title(f"Precision $\\uparrow$")
axs[3].set_title(f"Recall $\\uparrow$")


for ax in axs:
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Only show legend in the top-left subplot
leg = axs[1].legend(fontsize=4,fancybox=True, loc="lower right", mode="expand", bbox_to_anchor=(0.32, 0.03, 0.65, 0.6))

# Save the plot
fig.tight_layout()
plt.savefig('occ_learning.pgf')

