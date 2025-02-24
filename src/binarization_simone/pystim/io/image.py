def load(metadata):

    type_ = metadata['type']

    if type_ == 'iml':
        image = None
    elif type_ == 'png':
        raise NotImplementedError
    else:
        raise ValueError("unknown type value: {}".format(type_))

    return image
