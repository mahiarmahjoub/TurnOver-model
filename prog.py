# Argparser
# tutorial link: https://docs.python.org/2/howto/argparse.html
# here, we develop the code to take in 3 inputs for our main optimisation code 
# 'turnover.py': 
    # filename - name of the file containgin the fluorescence reads 
    # baseline - number of initial points taken to use its average to 
    # normalise data  
    # endpoint - number of endpoints taken to use the average to calculate 
    # relative fluorescence of the data 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="insert the name of file")
parser.add_argument("baseline", type=int,
                    help="# points to calculate baseline fluorescence")
parser.add_argument("endpoint", 
                    help="# points at the end to derive relative fluorescence",
                    type=int)
args = parser.parse_args()

print("filename is {} with {} and {} values".
      format(args.filename,args.baseline,args.endpoint))

# when merging with the main program, assign the arguments to program variables
# instead of printing them 