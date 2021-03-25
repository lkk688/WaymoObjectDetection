import pickle

filename = "/data/cmpe249-f20/WaymoKittitMulti/4c_train0002/waymo_dbinfos_train.pkl"
infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()

print(type(new_dict))
for key, value in new_dict.items() :
    #print (key, value)
    print(key)

# <class 'dict'>
# Car
# Pedestrian
# Sign
# Cyclist

# print(len(new_dict))
print(type(new_dict['Sign']))#<class 'list'>
print(len(new_dict['Sign']))
sign_dict = new_dict['Sign']

for i in range(15):
    print(sign_dict[i])


