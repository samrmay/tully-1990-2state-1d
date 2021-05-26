import csv
import matplotlib.pyplot as plt

FILES = {0: "results/05_19_21_v1_results/simple_0k_051921_2.csv",
         4.4: "results/05_19_21_v1_results/simple_4dot4k_051921.csv",
         4.5: "results/05_19_21_v1_results/simple_4dot5k_051921.csv",
         7.7: "results/05_19_21_v1_results/simple_7dot7k_051921.csv",
         7.8: "results/05_19_21_v1_results/simple_7dot8k_052021.csv",
         8.8: "results/05_19_21_v1_results/simple_8dot8k_051921.csv",
         10: "results/05_19_21_v1_results/simple_10k_052021.csv",
         12.5: "results/05_19_21_v1_results/simple_12dot5k_052021.csv",
         15: "results/05_19_21_v1_results/simple_15k_052021.csv",
         17.5: "results/05_19_21_v1_results/simple_17dot5k_052021.csv",
         20: "results/05_19_21_v1_results/simple_20k_052021.csv",
         25: "results/05_19_21_v1_results/simple_25k_052021.csv",
         27.5: "results/05_19_21_v1_results/simple_27dot5k_052021.csv",
         30: "results/05_19_21_v1_results/simple_30k_051921.csv",
         35: "results/05_19_21_v1_results/simple_35k_051921.csv"}

transmitted_state0 = {}
reflected_state0 = {}
transmitted_state1 = {}
x = FILES.keys()
for k in x:
    transmitted_state0[k] = 0
    reflected_state0[k] = 0
    transmitted_state1[k] = 0

    with open(FILES[k]) as file:
        file.readline()
        reader = csv.reader(file, delimiter=',')
        num = 0
        for row in reader:
            state = int(row[2])
            pos = float(row[0])

            if state == 0 and pos < 0:
                reflected_state0[k] += 1
            elif state == 0 and pos > 0:
                transmitted_state0[k] += 1
            elif state == 1 and pos > 0:
                transmitted_state1[k] += 1

            num += 1

        reflected_state0[k] /= num
        transmitted_state0[k] /= num
        transmitted_state1[k] /= num

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(x, reflected_state0.values())
ax1.plot(x, reflected_state0.values())
ax1.set_title("State 1 reflected")

ax2.scatter(x, transmitted_state0.values())
ax2.plot(x, transmitted_state0.values())
ax2.set_title("State 1 transmitted")

ax3.scatter(x, transmitted_state1.values())
ax3.plot(x, transmitted_state1.values())
ax3.set_title("State 2 transmitted")
plt.show()
