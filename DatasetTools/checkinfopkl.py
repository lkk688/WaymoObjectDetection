import pickle

filename = "/data/cmpe249-f20/WaymoKittitMulti/4c_train0002/waymo_infos_val.pkl"
infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()

print(type(new_dict))
for key, value in new_dict[0].items() :
    #print (key, value)
    print(key)
    
print(len(new_dict))
for i in range(len(new_dict)):
    print(new_dict[i]['annos']['name'])


