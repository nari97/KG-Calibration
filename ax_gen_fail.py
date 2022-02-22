import sys
import pickle

def run():
    #folder = sys.argv[1]
    folder = ""

    cases = ['transd_7_0']

    for c in cases:
        result = {}
        result['trial_index'] = int(c.split('_')[2])
        result_file = folder + "Ax/"+c+".fail"
        with open(result_file, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    run()