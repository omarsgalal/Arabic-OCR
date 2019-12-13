# files = ["errorsdict_0_2000.txt", "errorsdict_2000_4000.txt", "errorsdict_4000_all.txt"]
files = ["errorsdict_all_newdataset.txt"]
outfile = open("validimages_newdataset.txt", 'w')

for file in files:
    f = [line[:-1].split("    ") for line in open(file, 'r').readlines()[3:]]
    for item in f:
        if item[1] == "0":
            outfile.write(f"{item[0]}\n")

outfile.close()