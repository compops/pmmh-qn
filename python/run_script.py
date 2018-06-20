import sys
import scripts.example1_random_effects as example1
import scripts.helper_higgs as helper_higgs
import scripts.example2a_higgs as example2a
import scripts.example2b_higgs as example2b
import scripts.example3_stochastic_volatility as example3

higgs_data_path = '../data/higgs/'

if len(sys.argv) > 1:
    if (len(sys.argv) > 2) and int(sys.argv[2]) == 1:
        print("Running full experiment (5/10 Monte Carlo runs).")
        print("This will probably take a few hours.")
        NO_ITERS1 = 5
        NO_ITERS23 = 10
    else:
        print("Running reduced experiment (1 Monte Carlo run).")
        NO_ITERS1 = 1
        NO_ITERS23 = 1

    if int(sys.argv[1]) == 1:
        print("Running first example (random effects model).")
        for i in range(NO_ITERS1):
            example1.main(seed_offset=i)
    elif int(sys.argv[1]) == 2:
        print("Downloading and reformating Higgs data, this might take some time.")
        helper_higgs.get_data(file_path=higgs_data_path, subset=110000)

        print("Running second example (logistic regression model).")
        data = helper_higgs.load_data(file_path=higgs_data_path, subset=110000)
        for i in range(NO_ITERS23):
           example2a.main(data, use_all_data=False, seed_offset=i)
        for i in range(NO_ITERS23):
           example2b.main(data, use_all_data=False, seed_offset=i)
    elif int(sys.argv[1]) == 3:
        print("Running third example (stochastic volatility model).")
        for i in range(NO_ITERS23):
            example3.main(seed_offset=i)
    else:
        raise NameError("Unknown example.")
else:
    raise NameError("Need to provide the experiment to run.")
