from pqdm.processes import pqdm
# If you want threads instead:
# from pqdm.threads import pqdm

args = [1, 2, 3, 4, 5]
# args = range(1,6) would also work

def square(a):
    return a*a

result = pqdm(args, square, n_jobs=2)