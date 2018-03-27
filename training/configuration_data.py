
def get_label_info(dataset='panthera', type="empty"):

    if dataset == 'panthera':
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

        if type == 'empty':
            label_mapping = {'primary': {x: 'species' for x in
                keep_labels['primary'] if x not in ['vehicle', 'blank']}}
            label_mapping['primary']['vehicle'] = 'vehicle'
            label_mapping['primary']['blank'] = 'blank'

        elif type == 'species':
            keep_labels['primary'].remove('vehicle')
            keep_labels['primary'].remove('blank')

        else:
            raise NotImplementedError("type: %s not implemented" % type)

    else:
        raise NotImplementedError("dataset: %s not implemented" % dataset)

    res = {'labels_all': labels_all, 'keep_labels': keep_labels,
           'label_mapping': label_mapping}

    return res
