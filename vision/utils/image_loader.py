import io
import requests

from base64 import urlsafe_b64decode
from imageio import imread

from vision.utils.lang import lazy

def load_s3_unsigned ():
    import boto3
    from os import environ

    if 'UNSIGNED_AWS' in environ and environ['UNSIGNED_AWS'] == 'true':
        from botocore import UNSIGNED
        from botocore.client import Config
        return boto3.client('s3', config=Config(signature_version=UNSIGNED))
    else:
        return boto3.client('s3')

_s3 = lazy(lambda: load_s3_unsigned)

def image_from_urlsafe_base64 (image):
    return io.BytesIO(urlsafe_b64decode(image))

def image_from_s3 (image_s3, s3_client=None):
    if not s3_client:
        s3_client = _s3()

    if 'version' in image_s3:
        res = s3_client.get_object(
            Bucket=image_s3['bucket'],
            Key=image_s3['key'],
            VersionId=image_s3['version']
        )
    else:
        res = s3_client.get_object(
            Bucket=image_s3['bucket'],
            Key=image_s3['key']
        )
    body = res['Body'].read()
    return io.BytesIO(body)

def image_from_url (image_url):
    res = requests.get(image_url)
    return io.BytesIO(res.content)

def image_from_path (image_path):
    in_file = open(image_path, mode='rb')
    bytes = io.BytesIO(in_file.read())
    in_file.close()
    return bytes

def image_bytes_from_event (event, allow_fs=True, s3_client=None):
    if 'image_urlsafe_b64' in event:
        return image_from_urlsafe_base64(event['image_urlsafe_b64'])

    if 'image_s3' in event:
        return image_from_s3(event['image_s3'], s3_client)
    
    if 'image_url' in event:
        return image_from_url(event['image_url'])

    if not allow_fs:
        raise Exception('Missing input, use either image_url, image_urlsafe_b64, image_s3')

    if 'image_path' in event:
        return image_from_path(event['image_path'])

    raise Exception('Missing input, use either image_url, image_urlsafe_b64, image_s3, image_path')

def image_from_event (event, allow_fs=True, s3_client=None):
    return imread(image_bytes_from_event(event, allow_fs), s3_client)
