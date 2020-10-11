orders = ('Anseriformes', 'Apodiformes', 'Caprimulgiformes', 'Charadriiformes', 'Coraciiformes',
           'Cuculiformes', 'Gaviiformes', 'Passeriformes', 'Pelecaniformes', 'Piciformes',
           'Podicipediformes', 'Procellariiformes', 'Suliformes')

# order label for each species
order_labels = [11, 11, 11, 5, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 12, 12, 12, 7, 7, 7, 7, 7, 5, 5, 5, 7, 7, 9, 7, 7, 7, 7, 7, 7, 7, 12, 11, 0, 7, 7, 7, 10, 10, 10, 10, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 3, 7, 6, 0, 7, 0, 0, 7, 2, 7, 7, 7, 7, 7, 7, 7, 8, 8, 7, 7, 7, 2, 3, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7]
family_labels = []

curr = "train"
img_list_path = "../cvpr18-inaturalist-transfer/data/cub_200/images/{}.txt".format(curr)
f = open(img_list_path, "r")
img_list = f.readlines()
f.close()

# images.txt: 1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
# train.txt/val.txt: images/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg: 0

# out: 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg: 11

f.open("../cvpr18-inaturalist-transfer/data/cub_200/images/categorized_by_order_{}.txt".format(curr), "r")
labelled = []
for line in img_list:
    path, _ = line.split()
    species_idx = path[7:10]

    label = order_labels[int(species_idx)]
    out_string = path + ": " + str(label) + "\n"
    f.write(out_string)
f.close()
