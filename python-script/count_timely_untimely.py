import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

timely_count_new = 0
untimely_count_filled_new = 0
untimely_count_not_filled_new = 0
total_count_new = 0
incorrect_count_new = 0
oob_count_new = 0

filepath = Path(__file__).parent
rewards_big_spread = dict()
with open(f'{filepath}/result_10_-8.out', 'r') as f:
    for line in f:
        if "Prefetched Timely" in line:
           timely_count_new += 1
           total_count_new += 1
           nums = re.findall(r"\d+", line)
           delta = int(nums[1])
           reward = int(nums[2])
           rewards_big_spread[delta] = reward
        elif "Untimely evicted!" in line:
           untimely_count_not_filled_new += 1
           total_count_new += 1
        elif "Prefetched Untimely" in line:
           untimely_count_filled_new += 1
           total_count_new += 1
        elif "Incorrect evicted" in line:
            incorrect_count_new += 1
            total_count_new += 1
        elif "OOB evicted" in line:
            oob_count_new += 1
            total_count_new += 1

timely_count_base = 0
untimely_count_filled_base = 0
untimely_count_not_filled_base = 0
total_count_base = 0
incorrect_count_base = 0
oob_count_base = 0

with open(f'{filepath}/result_baseline_60_25.out', 'r') as f:
    for line in f:
        if "Prefetched Timely" in line:
            timely_count_base += 1
            total_count_base += 1
        elif "Untimely evicted!" in line:
            untimely_count_not_filled_base += 1
            total_count_base += 1
        elif "Prefetched Untimely" in line:
            untimely_count_filled_base += 1
            total_count_base += 1
        elif "Incorrect evicted" in line:
            incorrect_count_base += 1
            total_count_base += 1
        elif "OOB evicted" in line:
            oob_count_base += 1
            total_count_base += 1

rewards_small_spread = dict()
with open(f'{filepath}/result_7_-8.out', 'r') as f:
    for line in f:
        if "Prefetched Timely" in line:
            nums = re.findall(r"\d+", line)
            delta = int(nums[1])
            reward = int(nums[2])
            rewards_small_spread[delta] = reward
            
print(f"Timely: {timely_count_new} times, Untimely not filled: {untimely_count_not_filled_new} times, Untimely filled: {untimely_count_filled_new} times")
print(f"Timely Rate: {timely_count_new / (untimely_count_not_filled_new + untimely_count_filled_new)}")
print(f"Filled Rate {untimely_count_filled_new / untimely_count_not_filled_new}")
print(f"Total count: {total_count_new}")

cases_correct = ["Timely", "Untimely loaded", "Untimely unloaded", "Incorrect", "Out-of-Bounds", "Total prefetch count"]
fig, ax = plt.subplots(figsize=(10, 6))

counts = np.array([timely_count_new, untimely_count_filled_new, untimely_count_not_filled_new, incorrect_count_new, oob_count_new, total_count_new], dtype=np.int32)
baseline_counts = np.array([timely_count_base, untimely_count_filled_base, untimely_count_not_filled_base, incorrect_count_base, oob_count_base, total_count_base], dtype=np.int32)
x = np.arange(len(cases_correct))
width = 0.35
ax.bar(x - width/2, counts, width, label="Dynamic Rewards")
ax.bar(x + width/2, baseline_counts, width, label="Baseline")

ax.set_xticks(np.arange(len(cases_correct)))
plt.xticks(rotation=15, ha="right")
ax.ticklabel_format(style='plain', axis='y', scilimits=(0,8))
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, p: format(int(x), ','))
)
new_yticks = np.arange(0, max(counts)+1, 1_000_000)
ax.set_yticks(new_yticks)
ax.set_xticklabels(cases_correct)
ax.set_ylabel("Number of Occurrences")
ax.set_ylim(0,max(counts)*1.075)
ax.legend()

#plt.show()
plt.savefig(f"{filepath}/prefetches_count.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01)

fig2, ax2 = plt.subplots()

filtered_rewards = dict(filter(lambda z: z[0] < 50_000, rewards_small_spread.items()))
ax2.scatter(filtered_rewards.keys(), filtered_rewards.values(), label="Divisor of 128")

filtered_rewards = dict(filter(lambda z: z[0] < 50_000, rewards_big_spread.items()))
ax2.scatter(filtered_rewards.keys(), filtered_rewards.values(), label="Divisor of 1024")

ax2.set_xlabel("Time Delta (cycles)")
ax2.set_ylabel("Reward")
ax2.set_ylim(0, max(filtered_rewards.values())*1.075)
ax2.legend()

plt.savefig(f"{filepath}/rewards.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()