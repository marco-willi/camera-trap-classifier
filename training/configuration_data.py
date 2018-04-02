
def get_label_info(location='panthera', experiment="empty"):

    if dataset == 'west_africa':
        labels_all = {
          'species': [
            "Aardvark", "Antelope_Bates_Pygmy", "Baboon_Olive", "Badger_Honey",
            "Bat", "Bird", "Blank", "Bluebill_Red_Headed",
            "Bristlebill_Red_Tailed", "Buffalo_African", "Bushbuck", "Bushpig",
            "Cat_African_Golden", "Cat_Domestic", "Cattle_Domestic",
            "Chimpanzee_Eastern", "Civet_African", "Civet_African_Palm",
            "Colobus_Uganda_Red", "Cusimanse_Alexanders", "Dog_Domestic",
            "Duiker_Blue", "Duiker_Weyns", "Duiker_Yellow_Backed",
            "Elephant_African", "Fire", "Galago", "Genet_Rusty_Spotted",
            "Genet_Servaline", "Goat_Wild", "Guereza_Mantled",
            "Hog_Giant_Forest", "Human", "Hyaena_Spotted", "Jackal_Side_Striped",
            "Leopard_African", "Lion_African", "Mangabey_Grey_Cheeked",
            "Mongoose_Banded", "Mongoose_Egyptian", "Mongoose_Slender",
            "Mongoose_Water", "Mongouse_Banded", "Monitor_Nile", "Monkey_Blue",
            "Monkey_LHoests", "Monkey_Red_Tailed", "Monkey_Vervet",
            "Mouse_Congo_Forest", "Otter_Congo_Clawless",
            "Pangolin_African_White_Bellied", "Pangolin_Giant",
            "Porcupine_Crested", "Potto_Western", "Rat_Giant",
            "Rat_Greater_Cane", "Reptile", "Rodent", "Serval", "Sitatunga",
            "Snake", "Squirrel_African_Giant", "Tortoise_Common", "Vehicle",
            "Waterbuck"]}

        keep_labels = {
          'species': [
            "Aardvark", "Antelope_Bates_Pygmy", "Baboon_Olive", "Badger_Honey",
            "Bat", "Bird", "Blank", "Bluebill_Red_Headed",
            "Bristlebill_Red_Tailed", "Buffalo_African", "Bushbuck", "Bushpig",
            "Cat_African_Golden", "Cat_Domestic", "Cattle_Domestic",
            "Chimpanzee_Eastern", "Civet_African", "Civet_African_Palm",
            "Colobus_Uganda_Red", "Cusimanse_Alexanders", "Dog_Domestic",
            "Duiker_Blue", "Duiker_Weyns", "Duiker_Yellow_Backed",
            "Elephant_African", "Fire", "Galago", "Genet_Rusty_Spotted",
            "Genet_Servaline", "Goat_Wild", "Guereza_Mantled",
            "Hog_Giant_Forest", "Human", "Hyaena_Spotted", "Jackal_Side_Striped",
            "Leopard_African", "Lion_African", "Mangabey_Grey_Cheeked",
            "Mongoose_Banded", "Mongoose_Egyptian", "Mongoose_Slender",
            "Mongoose_Water", "Mongouse_Banded", "Monitor_Nile", "Monkey_Blue",
            "Monkey_LHoests", "Monkey_Red_Tailed", "Monkey_Vervet",
            "Mouse_Congo_Forest", "Otter_Congo_Clawless",
            "Pangolin_African_White_Bellied", "Pangolin_Giant",
            "Porcupine_Crested", "Potto_Western", "Rat_Giant",
            "Rat_Greater_Cane", "Reptile", "Rodent", "Serval", "Sitatunga",
            "Snake", "Squirrel_African_Giant", "Tortoise_Common", "Vehicle",
            "Waterbuck"]}

        label_mapping = None

        if experiment == 'empty':
            label_mapping = {'species': {x: 'Species' for x in
                keep_labels['species'] if x not in ['Vehicle', 'Blank']}}
            label_mapping['species']['Vehicle'] = 'Vehicle'
            label_mapping['species']['Blank'] = 'Blank'

        elif experiment == 'species':
            keep_labels['species'].remove('Vehicle')
            keep_labels['species'].remove('Blank')

        else:
            raise NotImplementedError("experiment: %s not implemented" % experiment)

    if location == 'southern_africa':
        labels_all = {
          'species': [
            "Aardvark", "Aardwolf", "Baboon_Chacma", "Badger_Honey", "Bat",
            "Bird", "Blank", "Blesbok", "Bontebok", "Buffalo_African",
            "Bushbaby", "Bushbaby_Greater", "Bushbaby_Lesser", "Bushbuck",
            "Bushpig", "Caracal", "Cheetah_African", "Civet_African",
            "Domestic_Animal", "Domestic_Cat", "Domestic_Cattle",
            "Domestic_Dog", "Domestic_Horse", "Donkey", "Duiker_Grey",
            "Duiker_Red", "Duiker_Steenbok", "Eland", "Elephant_African",
            "Fire", "Fox_Bat_Eared", "Fox_Cape", "Gemsbok",
            "Genet_Large_Spotted", "Genet_Small_Spotted", "Giraffe", "Goat",
            "Grysbok", "Hare_Jamesons_Red_Rock", "Hare_Scrub", "Hartebeest_Red",
            "Hartebeest_Tsessebe", "Hedgehog", "Hippopotamus", "Human",
            "Hyaena_Brown", "Hyaena_Spotted", "Hyrax_Rock", "Impala", "Insect",
            "Jackal_Black_Backed", "Jackal_Side_Striped", "Klipspringer",
            "Kudu_Greater", "Leopard_African", "Leopard_Tortoise",
            "Lion_African", "Mongoose", "Mongoose_Banded", "Mongoose_Dwarf",
            "Mongoose_Large_Grey", "Mongoose_Mellers", "Mongoose_Selous",
            "Mongoose_Slender", "Mongoose_Water", "Mongoose_White_Tailed",
            "Mongoose_Yellow", "Monkey", "Monkey_Samango", "Monkey_Vervet",
            "Nile_Monitor", "Nyala", "Ostrich", "Otter_Cape_Clawless",
            "Pangolin_Ground", "Porcupine_Cape", "Rabbit_Hare", "Rat_Cane",
            "Reedbuck_Common", "Reedbuck_Mountain", "Reedbuck_Rhebok",
            "Reptile", "Rhebok_Grey", "Rhino", "Rhino_Black", "Rhino_White",
            "Roan_Antelope", "Roan_Sable", "Rodent", "Sable_Antelope", "Serval",
            "Sheep", "Sitatunga", "Springbok",
            "Springhare", "Squirrel_Ground", "Squirrel_Tree", "Steenbok",
            "Suni", "Tsessebe", "Vehicle", "Warthog", "Waterbuck", "Wild_Dog",
            "Wildcat_African", "Wildebeest_Blue", "Zebra", "Zorilla"],
          'count_category': [
              "-1", "1", "2", "3", "4", "5", "6", "7",
              "8", "9", "10", "11-50", "51+"
            ]}

        keep_labels = {
          'species': [
            "Aardvark", "Aardwolf", "Baboon_Chacma", "Badger_Honey", "Bat",
            "Bird", "Blank", "Blesbok", "Bontebok", "Buffalo_African",
            "Bushbaby_Greater", "Bushbaby_Lesser", "Bushbuck",
            "Bushpig", "Caracal", "Cheetah_African", "Civet_African",
            "Domestic_Animal", "Domestic_Cat", "Domestic_Cattle",
            "Domestic_Dog", "Domestic_Horse", "Donkey", "Duiker_Grey",
            "Duiker_Red", "Eland", "Elephant_African",
            "Fire", "Fox_Bat_Eared", "Fox_Cape", "Gemsbok",
            "Genet_Large_Spotted", "Genet_Small_Spotted", "Giraffe", "Goat",
            "Grysbok", "Hare_Jamesons_Red_Rock", "Hare_Scrub", "Hartebeest_Red",
            "Hedgehog", "Hippopotamus", "Human",
            "Hyaena_Brown", "Hyaena_Spotted", "Hyrax_Rock", "Impala", "Insect",
            "Jackal_Black_Backed", "Jackal_Side_Striped", "Klipspringer",
            "Kudu_Greater", "Leopard_African", "Leopard_Tortoise",
            "Lion_African", "Mongoose_Banded", "Mongoose_Dwarf",
            "Mongoose_Large_Grey", "Mongoose_Mellers", "Mongoose_Selous",
            "Mongoose_Slender", "Mongoose_Water", "Mongoose_White_Tailed",
            "Mongoose_Yellow", "Monkey_Samango", "Monkey_Vervet",
            "Nile_Monitor", "Nyala", "Ostrich", "Otter_Cape_Clawless",
            "Pangolin_Ground", "Porcupine_Cape", "Rat_Cane",
            "Reedbuck_Common", "Reedbuck_Mountain",
            "Reptile", "Rhebok_Grey", "Rhino_Black", "Rhino_White",
            "Roan_Antelope", "Rodent", "Sable_Antelope", "Serval",
            "Sheep", "Sitatunga", "Springbok",
            "Springhare", "Squirrel_Ground", "Squirrel_Tree", "Steenbok",
            "Suni", "Tsessebe", "Vehicle", "Warthog", "Waterbuck", "Wild_Dog",
            "Wildcat_African", "Wildebeest_Blue", "Zebra", "Zorilla"],
          'count_category': [
              "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11-50", "51+"
            ]}

        label_mapping = None

        if experiment == 'empty':
            label_mapping = {'species': {x: 'Species' for x in
                keep_labels['species'] if x not in ['Vehicle', 'Blank']}}
            label_mapping['species']['Vehicle'] = 'Vehicle'
            label_mapping['species']['Blank'] = 'Blank'

        elif experiment == 'species':
            keep_labels['species'].remove('Vehicle')
            keep_labels['species'].remove('Blank')

        else:
            raise NotImplementedError("experiment: %s not implemented" % experiment)

    elif location == 'panthera':
        labels_all = {
          'primary': [
            'bat', 'hartebeest', 'insect', 'klipspringer', 'hyaenabrown',
            'domesticanimal', 'otter', 'hyaenaspotted', 'MACAQUE', 'aardvark',
            'reedbuck', 'waterbuck', 'bird', 'genet', 'blank', 'porcupine',
            'caracal', 'aardwolf', 'bushbaby', 'bushbuck', 'mongoose', 'polecat',
            'honeyBadger', 'reptile', 'cheetah', 'pangolin', 'giraffe', 'rodent',
            'leopard', 'roansable', 'hippopotamus', 'rabbithare', 'warthog', 'kudu',
            'batEaredFox', 'gemsbock', 'africancivet', 'rhino', 'wildebeest',
            'monkeybaboon', 'zebra', 'bushpig', 'elephant', 'nyala', 'jackal',
            'serval', 'buffalo', 'vehicle', 'eland', 'impala', 'lion',
            'wilddog', 'duikersteenbok', 'HUMAN', 'wildcat']}

        keep_labels = {
          'primary': [
            'bat', 'hartebeest', 'insect', 'klipspringer', 'hyaenabrown',
            'domesticanimal', 'hyaenaspotted', 'aardvark',
            'reedbuck', 'waterbuck', 'bird', 'genet', 'blank', 'porcupine',
            'caracal', 'aardwolf', 'bushbaby', 'bushbuck', 'mongoose',
            'honeyBadger', 'cheetah', 'giraffe', 'rodent',
            'leopard', 'roansable', 'hippopotamus', 'rabbithare', 'warthog', 'kudu',
            'batEaredFox', 'gemsbock', 'africancivet', 'rhino', 'wildebeest',
            'monkeybaboon', 'zebra', 'bushpig', 'elephant', 'nyala', 'jackal',
            'serval', 'buffalo', 'vehicle', 'eland', 'impala', 'lion',
            'wilddog', 'duikersteenbok', 'HUMAN', 'wildcat']}

        label_mapping = None

        if experiment == 'empty':
            label_mapping = {'primary': {x: 'species' for x in
                keep_labels['primary'] if x not in ['vehicle', 'blank']}}
            label_mapping['primary']['vehicle'] = 'vehicle'
            label_mapping['primary']['blank'] = 'blank'

        elif experiment == 'species':
            keep_labels['primary'].remove('vehicle')
            keep_labels['primary'].remove('blank')

        else:
            raise NotImplementedError("experiment: %s not implemented" % experiment)

    else:
        raise NotImplementedError("location: %s not implemented" % location)

    res = {'labels_all': labels_all, 'keep_labels': keep_labels,
           'label_mapping': label_mapping}

    return res



























def get_label_info(dataset='panthera', type="empty"):

    if dataset == 'panthera':
        labels_all = {
          'species': [
            "Aardvark", "Antelope_Bates_Pygmy", "Baboon_Olive", "Badger_Honey",
            "Bat", "Bird", "Blank", "Bluebill_Red_Headed",
            "Bristlebill_Red_Tailed", "Buffalo_African", "Bushbuck", "Bushpig",
            "Cat_African_Golden", "Cat_Domestic", "Cattle_Domestic",
            "Chimpanzee_Eastern", "Civet_African", "Civet_African_Palm",
            "Colobus_Uganda_Red", "Cusimanse_Alexanders", "Dog_Domestic",
            "Duiker_Blue", "Duiker_Weyns", "Duiker_Yellow_Backed",
            "Elephant_African", "Fire", "Galago", "Genet_Rusty_Spotted",
            "Genet_Servaline", "Goat_Wild", "Guereza_Mantled",
            "Hog_Giant_Forest", "Human", "Hyaena_Spotted", "Jackal_Side_Striped",
            "Leopard_African", "Lion_African", "Mangabey_Grey_Cheeked",
            "Mongoose_Banded", "Mongoose_Egyptian", "Mongoose_Slender",
            "Mongoose_Water", "Mongouse_Banded", "Monitor_Nile", "Monkey_Blue",
            "Monkey_LHoests", "Monkey_Red_Tailed", "Monkey_Vervet",
            "Mouse_Congo_Forest", "Otter_Congo_Clawless",
            "Pangolin_African_White_Bellied", "Pangolin_Giant",
            "Porcupine_Crested", "Potto_Western", "Rat_Giant",
            "Rat_Greater_Cane", "Reptile", "Rodent", "Serval", "Sitatunga",
            "Snake", "Squirrel_African_Giant", "Tortoise_Common", "Vehicle",
            "Waterbuck"],
          'count': [

            ]}

        keep_labels = {
          'species': [
            "Aardvark", "Antelope_Bates_Pygmy", "Baboon_Olive", "Badger_Honey",
            "Bat", "Bird", "Blank", "Bluebill_Red_Headed",
            "Bristlebill_Red_Tailed", "Buffalo_African", "Bushbuck", "Bushpig",
            "Cat_African_Golden", "Cat_Domestic", "Cattle_Domestic",
            "Chimpanzee_Eastern", "Civet_African", "Civet_African_Palm",
            "Colobus_Uganda_Red", "Cusimanse_Alexanders", "Dog_Domestic",
            "Duiker_Blue", "Duiker_Weyns", "Duiker_Yellow_Backed",
            "Elephant_African", "Fire", "Galago", "Genet_Rusty_Spotted",
            "Genet_Servaline", "Goat_Wild", "Guereza_Mantled",
            "Hog_Giant_Forest", "Human", "Hyaena_Spotted", "Jackal_Side_Striped",
            "Leopard_African", "Lion_African", "Mangabey_Grey_Cheeked",
            "Mongoose_Banded", "Mongoose_Egyptian", "Mongoose_Slender",
            "Mongoose_Water", "Mongouse_Banded", "Monitor_Nile", "Monkey_Blue",
            "Monkey_LHoests", "Monkey_Red_Tailed", "Monkey_Vervet",
            "Mouse_Congo_Forest", "Otter_Congo_Clawless",
            "Pangolin_African_White_Bellied", "Pangolin_Giant",
            "Porcupine_Crested", "Potto_Western", "Rat_Giant",
            "Rat_Greater_Cane", "Reptile", "Rodent", "Serval", "Sitatunga",
            "Snake", "Squirrel_African_Giant", "Tortoise_Common", "Vehicle",
            "Waterbuck"],
            'count': [

            ]}

        label_mapping = None

        if type == 'empty':
            label_mapping = {'species': {x: 'species' for x in
                keep_labels['species'] if x not in ['vehicle', 'blank']}}
            label_mapping['species']['vehicle'] = 'vehicle'
            label_mapping['species']['blank'] = 'blank'

        elif type == 'species':
            keep_labels['species'].remove('vehicle')
            keep_labels['species'].remove('blank')

        else:
            raise NotImplementedError("type: %s not implemented" % type)

    else:
        raise NotImplementedError("dataset: %s not implemented" % dataset)

    res = {'labels_all': labels_all, 'keep_labels': keep_labels,
           'label_mapping': label_mapping}

    return res
