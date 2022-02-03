from unittest.case import TestCase
from vision.utils import image_loader
from base64 import b64decode, urlsafe_b64encode, b64encode

b64 = b'iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAAQMAAABmvDolAAAAA1BMVEW10NBjBBbqAAAAH0lEQVRoge3BAQ0AAADCoPdPbQ43oAAAAAAAAAAAvg0hAAABmmDh1QAAAABJRU5ErkJggg=='
events = [
    { 'image_url': 'https://web.archive.org/web/20210227090658im_/https://www.mjt.me.uk/assets/images/smallest-png/openstreetmap.png' },
    { 'image_path': './test_images/small.png' },
    { 'image_urlsafe_b64': urlsafe_b64encode(b64decode(b64)) },
    { 'image_s3':
        {
            'bucket': 'kycdeepface-test',
            'key': 'small.png'
        }
    }
]

class ImageLoaderTest(TestCase):

    def test_events(self):
        for event in events:
            image = image_loader.image_bytes_from_event(event)
            res = b64encode(image.read())
            self.assertEqual(
                res,
                b64
            )
