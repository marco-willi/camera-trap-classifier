from PIL import Image
import io

sample_image = 'D:\Studium_GD\Zooniverse\SnapshotSafari\data\deploy\sample_ele.jpeg'

img = Image.open(sample_image)
b = io.BytesIO()
img.save(b, 'JPEG')
image_bytes = b.getvalue()


img2 = Image.open(sample_image)
b2 = io.BytesIO()
image_bytes2 = b2.getvalue()


image_url = "https://panoptes-uploads.zooniverse.org/production/subject_location/9e4556a3-5aba-46d0-a932-1f0e9e158d0d.jpeg"
import urllib.request as req
with req.urlopen(image_url) as url:
    b3 = io.BytesIO(url.read())
    image = b3.getvalue()
