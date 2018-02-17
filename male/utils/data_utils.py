from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import sys
import shutil
import hashlib
import tarfile
# from six.moves.urllib.request import urlopen
# from six.moves.urllib.error import URLError, HTTPError
# import six.moves.urllib as urllib
# from urllib.request import urlopen
from urllib.error import URLError, HTTPError
# import urllib
import requests

# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
# if sys.version_info[0] == 2:


def urlretrieve(url, filename, reporthook=None, data=None, total_size=None):
    def chunk_read(response, chunk_size=8192, reporthook=None, total_size=None):
        if total_size is None:
            if 'Content-length' in response.headers.keys():
                total_size = int(response.headers['content-length'])
            else:
                total_size = 0
        with open(filename, 'wb') as fd:
            count = 0
            for chunk in response.iter_content(chunk_size):
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                fd.write(chunk)

    s = requests.Session()
    res1 = s.get(url, stream=True)

    chunk_read(res1, reporthook=reporthook, total_size=total_size)
    # print(os.stat(filename))
    if os.stat(filename).st_size < 10000:
        print('Get new link ...')
        with open(filename, 'r') as my_file:
            data = my_file.read()
        result = re.search('(confirm=.*?)(&)', data)
        confirm = result.group(1)
        url += '&' + confirm

        # Use the cookie is subsequent requests
        res2 = s.get(url, stream=True)
        chunk_read(res2, reporthook=reporthook, total_size=total_size)


def get_file(fname, origin, untar=False,
             md5_hash=None, cache_subdir='', total_size=None):
    """Downloads a file from a URL if it not already in the cache.

    Passing the MD5 hash will verify the file after download
    as well as if it is already present in the cache.

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache

    # Returns
        Path to the downloaded file
    """
    from ..configs import data_dir
    from .generic_utils import Progbar

    datadir = os.path.join(data_dir(), cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size, show_steps=1)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress, total_size=total_size)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None
        print()

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def data_info(dataset):
    a9a = dict()
    a9a['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1U5id1yyUOB9eU6re3PKvULJ0Dww1Cpsz&export=download'
    a9a['md5_hash'] = '0dda1a6f19e3b51c9f2251cdc89f8a41'
    a9a['size'] = None

    poker = dict()
    poker['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-5dTSIJGdmyZ9bmVUwFNin9kWOpvqxjU&export=download'
    poker['md5_hash'] = '7733edcb1b674b246c35c1e01f09a8e7'
    poker['size'] = 9000000

    w8a = dict()
    w8a['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-8_DR6XryBMR0As71cQjSC-p1Fk-Qf-Q&export=download'
    w8a['md5_hash'] = '74e119dafbc8a4f1fedf2a7745cb4e9f'
    w8a['size'] = 3500000

    codrna = dict()
    codrna['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1Rq3XEY89-QUJ3777vUsyKss0Rd5ax9Ux&export=download'
    codrna['md5_hash'] = '74e119dafbc8a4f1fedf2a7745cb4e9f'
    codrna['size'] = 13000000

    ijcnn1 = dict()
    ijcnn1['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-AJUeM7x_uQq_pOKkSCmft3IGRigaiJz&export=download'
    ijcnn1['md5_hash'] = '3a8cae3e5769a728e7561c3b86201a0b'
    ijcnn1['size'] = 13000000

    seismic = dict()
    seismic['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-F4RicZZAelnH4Eucarq-KU07dYhdUB2&export=download'
    seismic['md5_hash'] = '494ee113f23a46ae7a057e96b348ec92'
    seismic['size'] = 110000000

    connect4 = dict()
    connect4['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-BWjSSHmY0qKeGIelv8fjJJT8jWIGRBk&export=download'
    connect4['md5_hash'] = 'c307a15326fe633cb0214453b02079ef'
    connect4['size'] = 5000000

    skin = dict()
    skin['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-M6Kv-fRiuXZ9qdSg9gm9D7YaN_SkonB&export=download'
    skin['md5_hash'] = '40a133c6a2b22626367466f2f4c60fac'
    skin['size'] = 7000000

    epsilon = dict()
    epsilon['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1R72Kq79hfzFaRVlxygBcu9VHCtVO9ytM&export=download'
    epsilon['md5_hash'] = 'd38b00d80497a282e1e4bd28177ea7dc'
    epsilon['size'] = 7600000000

    covtype = dict()
    covtype['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1TRMK-mfAohzRZn8bGijvMnBS4shQBYhI&export=download'
    covtype['md5_hash'] = 'c584928b0a3cec2034cd8b39095bdfcd'
    covtype['size'] = 23000000

    susy = dict()
    susy['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1QpPcWh5o8bOuL-RpNvouwr_a_kusP7bq&export=download'
    susy['md5_hash'] = '529408b2c9ae50c42e1d15a7fa3b121e'
    susy['size'] = 1300000000

    higgs = dict()
    higgs['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-SrT7NgikazgZUskQtNKlIoOtuvcuVgm&export=download'
    higgs['md5_hash'] = 'e13fd8d331b9ff2819b8d46603e16508'
    higgs['size'] = 3600000000

    info = dict()
    info['a9a'] = a9a
    info['poker'] = poker
    info['w8a'] = w8a
    info['cod-rna'] = codrna
    info['ijcnn1'] = ijcnn1
    info['seismic'] = seismic
    info['connect-4'] = connect4
    info['skin'] = skin
    info['epsilon'] = epsilon
    info['covtype'] = covtype
    info['susy'] = susy
    info['higgs'] = higgs
    return info[dataset]


def validate_file(fpath, md5_hash):
    '''Validates a file against a MD5 hash

    # Arguments
        fpath: path to the file being validated
        md5_hash: the MD5 hash being validated against

    # Returns
        Whether the file is valid
    '''
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


def scp_file(filepath):
    pass


if __name__ == '__main__':
    scp_file(sys.argv[1])
