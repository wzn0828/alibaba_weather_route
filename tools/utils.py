import os


def HMS(sec):
    '''
    :param sec: seconds
    :return: print of H:M:S
    '''

    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)

    return "%dh:%02dm:%02ds" % (h, m, s)


def configurationPATH(cf):
    '''
    :param cf: config file
    :return: Print some paths
    '''

    print("\n###########################")
    print(' > Save Path = "%s"' % (cf.savepath))
    print(' > Dataset PATH = "%s"' % (os.path.join(cf.dataroot_dir)))
    print("###########################\n")
