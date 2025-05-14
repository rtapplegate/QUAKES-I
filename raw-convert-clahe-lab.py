#!/usr/bin/python3

r'''Prepate QUAKES data for Pix4D

This is the pre-processing tool used to convert the data we collect into a form
that's ingestable by Pix4D. The inputs are:

- The images. These may be something standard (like .jpg) or "raw" data. Here
  "raw" means some opaque header we ignore, followed by the imager output, 8
  bits per pixel. The output may be debayered to produce color data, and some image
  processing may be applied to nice-ify the output

- INS data from the airplane. A .vnl table of plane poses, synced with the frame
  numbers in the captured images. This is optional. If omitted, no pose data is
  written into the EXIF in the output

- Camera metadata: model numbers and serial numbers of the cameras. Optional. If
  omitted, we don't write it

- Plane-camera transform. These are represented by 3 euler angles. If omitted,
  no pose information is written into the EXIF in the output

- Geometric correction. A full 6DOF transform applied to the camera poses, to
  compensate for errors in the other geometric parameters. If omitted, no
  correction is applied.

- Camera models: a mrcal model used to provide the intrinsics. If given, these
  are written into the EXIF of the output

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    def comma_separated_list(string, N, t):
        try:
            l = [t(x) for x in string.split(',')]
        except:
            raise argparse.ArgumentTypeError(f"Couldn't parse '{string}' as a comma-separated list of values of type {t}")
        if len(l) != N:
            raise argparse.ArgumentTypeError(f"The argument'{string}' should have {N} values, but instead it had {len(l)} values")
        return l

    parser.add_argument('--body-manufacturer',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Image.Make' tag. If omitted, we will not write
                        this tag''')
    parser.add_argument('--body-model',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Image.Model' tag. If omitted, we will not write
                        this tag''')
    parser.add_argument('--lens-model',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.LensModel' tag. If omitted, we will not
                        write this tag''')
    parser.add_argument('--body-serial-number',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.BodySerialNumber' tag. If omitted, we will
                        not write this tag''')
    parser.add_argument('--lens-serial-number',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.LensSerialNumber' tag. If omitted, we will
                        not write this tag''')
    parser.add_argument('--serial-number',
                        type = str,
                        help=r'''Arbitrary additional string to be written into
                        the "lens model". This primarily exists for pix4d's
                        benefit. Pix4d displays the body and lens models, but no
                        serial numbers. So this is a hack to make the serial
                        number show up in pix4d. If omitted, we will not write
                        this tag''')
    parser.add_argument('--focal-length-nominal-mm',
                        type = float,
                        help=r'''Arbitrary value to be written into the
                        'Exif.Photo.FocalLength' tag. If omitted, we will not
                        write this tag. This is the nominal focal length from
                        the lens manufacturer. The calibrated focal lengths come
                        from the camera model''')
    parser.add_argument('--pixels-per-mm',
                        type = float,
                        help=r'''The pixel pitch of the sensor. If given, this
                        is written into the 'Exif.Photo.FocalPlaneXResolution'
                        and 'Exif.Photo.FocalPlaneYResolution' tags. And it is
                        used to scale the focal length and principal point tags
                        in the EXIF, since those are given in mm. The actual
                        value of this tag doesn't matter: when the downstream
                        tool reads these tags, it will convert the focal lengths
                        and principal point back to units of pixels. But for
                        good hygiene, do provide this value. If omitted, we will
                        not write any tags that need this data, so --model must
                        be omitted also. Similarly, if given, --model must be
                        given also.''')

    parser.add_argument('--metadata',
                        type=str,
                        help='''A vnl containing a time-series of poses of the
                        airplane or cameras. Each row represents an image
                        captured from a particular pose. Exclusive with any
                        image globs passed in on the commandline''')

    parser.add_argument('--camera-roll-deg',
                        type = float,
                        default = 0,
                        help='''Used only if --metadata contains world-airplane
                        transformations. The camera roll, relative to the
                        airplane, in degrees. 0deg = "camera is mounted
                        normally"; 90deg = "camera is mounted sideways"''')
    parser.add_argument('--camera-yaw-deg',
                        type = float,
                        default = 0,
                        help='''Used only if --metadata contains world-airplane
                        transformations. The camera yaw, relative to the
                        airplane, in degrees. 0deg = "camera is looking in the
                        forward airplane direction"; 90deg = "camera is looking
                        to the left of the airplane"''')
    parser.add_argument('--camera-pitch-deg',
                        type = float,
                        default = 0,
                        help='''Used only if --metadata contains world-airplane
                        transformations. The camera pitch, relative to the
                        airplane, in degrees. 0deg = "camera is looking along
                        the horizontal"; 90deg = "camera is looking straight
                        down"''')
    parser.add_argument('--camera-pitch-then-yaw',
                        action='store_true',
                        help='''Used only if --metadata contains world-airplane
                        transformations. Specifies whether --camera-pitch-deg is
                        applied before --camera-yaw-deg, or the other way
                        around. By default, we yaw and then pitch (used for the
                        mostly-side-looking SAR fusion geometry). With
                        --camera-pitch-then-yaw we go the other way (used for
                        the mostly down-looking QUAKES geometry)''')
    parser.add_argument('--rt-world-new-world-old',
                        type = lambda s: comma_separated_list(s,6,float),
                        help='''A correction to apply to the world <-> camera
                        transformations. This is a mrcal rt transform (6 values)
                        that maps the original world coord to a corrected one.
                        Must be given as a comma-separated list''')

    parser.add_argument('--rt-cam-new-cam-old',
                        type = lambda s: comma_separated_list(s,6,float),
                        help='''A correction to apply to the world <-> camera
                        transformations. This is a mrcal rt transform (6 values)
                        that maps the original camera coords to corrected ones.
                        Must be given as a comma-separated list''')

    parser.add_argument('--image-path-prefix',
                        help='''If given, we prepend the given prefix to the
                        image paths in the metadata. Exclusive with
                        --image-directory''')

    parser.add_argument('--image-directory',
                        help='''If given, we look at the filenames (not
                        directories) from the metadata only, and we search THIS
                        directory. Exclusive with --image-path-prefix''')

    parser.add_argument('--model',
                        type=str,
                        help='''A cameramodel describing the camera used to
                        capture the given images. We write the intrinsics into
                        the EXIF tags. If omitted, we do not write this
                        information into the EXIF tags. Only LENSMODEL_PINHOLE,
                        LENSMODEL_OPENCV4 and LENSMODEL_OPENCV5 are supported''')

    parser.add_argument('--image-dims',
                        type = lambda s: comma_separated_list(s,2,int),
                        help='''A "WIDTH,HEIGHT" pair to specify the image size.
                        This is used only if we're given images that require a
                        raw conversion. This is REQUIRED for raw conversions''')

    parser.add_argument('--outdir',
                        type=str,
                        required=True,
                        help='''The directory where the output images should be
                        written. Must be specified, and must NOT refer to the
                        input images. This tool may overwrite some files, but it
                        will NEVER overwrite its input''')

    parser.add_argument('--grayworld',
                        action = 'store_true',
                        help='''Whether we apply a color rebalancing or not. By
                        default we leave the colors as they are''')

    parser.add_argument('--saturate-recip',
                        type    = int,
                        default = 65536,
                        help='''We apply a histogram stretch when processing the
                        raw images. To produce visually-pleasing results, we
                        sacrifice a small amount of the histogram at the low and
                        high ends. This produces good contrast in the bulk of
                        the image. The sacrificial amount is specified by this
                        argument. For instance "--saturate-recip 100" means the
                        darkest 1/100th of the pixels end up at 0, and the
                        brightest 1/100th end up at 255. In practice, a much
                        bigger value than 100 is desired. We default to 65536''')

    parser.add_argument('--camera-poses-only',
                        action = 'store_true',
                        help='''If given, we ONLY produce the vnl with the
                        camera poses. We do NOT do any raw conversion. We do NOT
                        write any images. We do NOT write EXIF tags to
                        anything''')

    parser.add_argument('--jobs', '-j',
                        type=int,
                        default=1,
                        help='''parallelize the processing JOBS-ways. This is
                        like Make, except you're required to explicitly specify
                        a job count. If omitted, no parallelization happens''')

    parser.add_argument('--prototype-raw-converter',
                        action='store_true',
                        help='''Uses the slow written-in-python implementation
                        of the raw converter. For debugging''')

    parser.add_argument('images',
                        type=str,
                        nargs='*',
                        help='''The images we're processing. Optional. Exclusive
                        with --metadata. world-camera transform must be present
                        in the EXIF; this will be repacked into other EXIF tags
                        for pix4d. If the images is already in a standard
                        format, I dont mess with the image bits: I leave them
                        alone''')

    args = parser.parse_args()

    if args.metadata is not None and \
       len(args.images) > 0:
        print("Exactly one of --metadata and images must be passed-in. Here we have both",
              file=sys.stderr)
        sys.exit(1)
    if args.metadata is None and \
       len(args.images) == 0:
        print("Exactly one of --metadata and images must be passed-in. Here we have none",
              file=sys.stderr)
        sys.exit(1)

    if args.image_path_prefix is not None and \
       args.image_directory is not None:
        print("--image-path-prefix and --image-directory are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)

    return args

args = parse_args()




import numpy as np
import numpysane as nps
import glob
import pyproj
import pyexiv2
import fractions
import mrcal
import vnlog
import cv2
import shutil
import multiprocessing
import signal

sys.path[:0] = '',
from generic_functions import parse_filenames_poses_from_metadata
from generic_functions import parse_filenames_poses_from_exif
from generic_functions import ne_from_latlon
from generic_functions import latlon_from_ne


if not args.prototype_raw_converter:
    import _raw_convert


if args.model is not None:
    try:
        m = mrcal.cameramodel(args.model)
    except:
        print(f"Couldn't parse cameramodel from '{args.model}'", file=sys.stderr)
        sys.exit(1)

    lensmodel = m.intrinsics()[0]
    if not (lensmodel == 'LENSMODEL_PINHOLE' or
            lensmodel == 'LENSMODEL_OPENCV4' or
            lensmodel == 'LENSMODEL_OPENCV5'):
        print(f"I only support LENSMODEL_PINHOLE, LENSMODEL_OPENCV4, LENSMODEL_OPENCV5, but read '{lensmodel}' from '{args.model}'", file=sys.stderr)
        sys.exit(1)

    if args.pixels_per_mm is None:
        print(f"Both --model and --pixels-per-mm may not be given",
              file=sys.stderr)
        sys.exit(1)

else:
    if args.pixels_per_mm is not None:
        print(f"Both --model and --pixels-per-mm should be given, or neither should be", file=sys.stderr)
        sys.exit(1)

def compute__latlonalt_opkdeg(Rt_world_cam):

    r'''Produce pix4d-style poses'''

    # "world" is (N,E,down)

    latlonalt_opkdeg = np.zeros( Rt_world_cam.shape[:-2] + (6,), dtype=float)

    ned = Rt_world_cam[...,3,:]

    latlonalt_opkdeg[..., :2] = latlon_from_ne(ned[..., :2])

    # altitude is measured UP, so I invert the DOWN value
    latlonalt_opkdeg[..., 2] = -ned[..., 2]

    # Docs are here:
    #
    #   https://support.pix4d.com/hc/en-us/articles/202558969

    R_world_cam = Rt_world_cam[..., :3,:]

    # The pdf document above describes a transformation C_EB.
    #
    # I'm assuming the Earth is locally flat, so the coord system E is always
    # aligned with the ground. Docs say this coord system is (E,N,up)
    R_E_world = np.array(((0,1, 0),
                          (1,0, 0),
                          (0,0,-1)))

    # The coord system B is aligned with the camera. It is (right,up,back)
    R_cam_B = np.array(((1, 0, 0),
                        (0,-1, 0),
                        (0, 0,-1)))

    R_EB = nps.matmult(R_E_world, R_world_cam, R_cam_B)

    # The pdf document defines C_EB (same as my R_EB) as a function of
    # (omega,phi,kappa). I decompose

    # I assume that phi is in [-180,180], so cos(phi)>=0
    sp = R_EB[..., 0,2]
    p = np.arcsin(sp)
    k = np.arctan2(-R_EB[..., 0,1], R_EB[..., 0,0])
    o = np.arctan2(-R_EB[..., 1,2], R_EB[..., 2,2])

    latlonalt_opkdeg[..., 3] = o*180./np.pi
    latlonalt_opkdeg[..., 4] = p*180./np.pi
    latlonalt_opkdeg[..., 5] = k*180./np.pi
    return latlonalt_opkdeg

def process_image(filename,

                  # Anything that is None is not set
                  latlonalt_opkdeg        = None,
                  model_intrinsics        = None,
                  body_manufacturer       = None,
                  body_model              = None,
                  lens_model              = None,
                  body_serial_number      = None,
                  lens_serial_number      = None,
                  serial_number           = None,
                  focal_length_nominal_mm = None,
                  pixels_per_mm           = None):

    def rational(x, denominator=65536):
        numerator = int(x*denominator)
        return fractions.Fraction(numerator,denominator)
    def GPSCoordinate(x):
        ax = abs(x)
        d  = int(ax)
        m  = int(60*(ax - d))
        s  = 3600*(ax - d) - 60*m
        return \
            [ fractions.Fraction(d,1),
              fractions.Fraction(m,1),
              rational(s) ]


    tags = dict()

    if model_intrinsics is not None:
        lensmodel, intrinsics = model_intrinsics.intrinsics()
        if not (lensmodel == 'LENSMODEL_PINHOLE' or
                lensmodel == 'LENSMODEL_OPENCV4' or
                lensmodel == 'LENSMODEL_OPENCV5'):
            raise Exception("I only support LENSMODEL_{PINHOLE,OPENCV4,OPENCV5}")

        fxy = intrinsics[0:2]
        cxy = intrinsics[2:4]
        if lensmodel == 'LENSMODEL_PINHOLE':
            r1,r2,t1,t2,r3 = 0,0,0,0,0
        elif lensmodel == 'LENSMODEL_OPENCV4':
            r1,r2,t1,t2 = intrinsics[4:]
            r3 = 0
        elif lensmodel == 'LENSMODEL_OPENCV5':
            r1,r2,t1,t2,r3 = intrinsics[4:]

        tags['Xmp.Camera.PerspectiveDistortion'] = ','.join(str(x) for x in (r1,r2,r3,t1,t2))

        if pixels_per_mm is not None:

            fxy_mean = (fxy[0] + fxy[1])/2.

            if np.abs(fxy[0] - fxy[1]) > 1e-6:
                print(f"## WARNING: Pix4d assumes fx==fy, but this model has unequal fxy: {fxy}. I write the mean into the EXIF: {fxy_mean}")

            tags['Xmp.Camera.PrincipalPoint']         = f'{cxy[0]/pixels_per_mm},{cxy[1]/pixels_per_mm}'
            tags['Xmp.Camera.PerspectiveFocalLength'] = str(fxy_mean / pixels_per_mm)
            tags['Exif.Photo.FocalPlaneXResolution']  = rational(pixels_per_mm)
            tags['Exif.Photo.FocalPlaneYResolution']  = rational(pixels_per_mm)

    if body_manufacturer       is not None: tags['Exif.Image.Make']             = body_manufacturer
    if body_model              is not None: tags['Exif.Image.Model']            = body_model
    if lens_model              is not None: tags['Exif.Photo.LensModel']        = lens_model
    if body_serial_number      is not None: tags['Exif.Photo.BodySerialNumber'] = body_serial_number
    if lens_serial_number      is not None: tags['Exif.Photo.LensSerialNumber'] = lens_serial_number
    if focal_length_nominal_mm is not None: tags['Exif.Photo.FocalLength']      = rational(focal_length_nominal_mm)

    if serial_number is not None:
        if 'Exif.Photo.LensModel' not in tags:
            tags['Exif.Photo.LensModel'] = serial_number
        else:
            tags['Exif.Photo.LensModel'] += ': ' + serial_number

    if latlonalt_opkdeg is not None:
        lat,lon,alt = latlonalt_opkdeg[:3]

        tags['Exif.GPSInfo.GPSLatitude']     = GPSCoordinate(abs(lat))
        tags['Exif.GPSInfo.GPSLongitude']    = GPSCoordinate(abs(lon))
        tags['Exif.GPSInfo.GPSAltitude']     = rational(alt)
        tags['Exif.GPSInfo.GPSLatitudeRef']  = 'N' if lat >= 0 else 'S'
        tags['Exif.GPSInfo.GPSLongitudeRef'] = 'E' if lon >= 0 else 'W'
        tags['Exif.GPSInfo.GPSAltitudeRef']  = '0' # above sea level

        # Indicate very poor GPS accuracy
        tags['Xmp.Camera.GPSXYAccuracy']     = "1000"
        tags['Xmp.Camera.GPSZAccuracy']      = "1000"


    # Split the filename into = d + f + e (directory, basename, extension)
    d,f = os.path.split(filename)
    f,e = os.path.splitext(f)

    if is_raw_image(filename):
        # I convert the raw, write the .jpg to disk, and then update the exif of
        # the file on disk
        if args.prototype_raw_converter:
            image = raw_convert__prototype(filename,
                                           *args.image_dims,
                                           unbayer = True)

        else:
            image = _raw_convert.raw_convert(filename,
                                             *args.image_dims,
                                             do_grayworld   = args.grayworld,
                                             saturate_recip = args.saturate_recip)

        filename = f"{args.outdir}/{f}.jpg"

        try:
            cv2.imwrite(filename,
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        except:
            msg = f"Couldn't imwrite('{filename}')"
            # map_async swallows errors, so I print to the console also
            print(msg, file=sys.stderr)
            raise Exception(msg)

    else:
        filename_out = f"{args.outdir}/{f}{e}"
        shutil.copy(filename, filename_out)
        filename = filename_out

    if not tags:
        # no EXIF tags to write. I'm done
        return filename

    exif = pyexiv2.ImageMetadata(filename)
    exif.read()

    # Required to be able to touch Xmp.Camera tags
    try:
        pyexiv2.register_namespace('https://support.pix4d.com/hc/en-us/articles/205732309-EXIF-and-XMP-tag-information-read-by-Pix4D-Desktop/', 'Camera')
    except:
        pass

    tags.update({ 'Xmp.Camera.ModelType':                'perspective',
                  'Exif.Photo.FocalPlaneResolutionUnit': 4, # "mm"
                 })

    for key in tags:
        if key in exif:
            print(f"## WARNING: key '{key}' already exists in the metadata for '{filename}'. Modifying existing tags is an untested case, so I do nothing for now")
            continue

        if   re.match('Xmp',  key): tag = pyexiv2.XmpTag (key, tags[key])
        elif re.match('Exif', key): tag = pyexiv2.ExifTag(key, tags[key])
        else: raise Exception("I only know how to make EXIF and XMP tags")
        exif[key] = tag

    try:
        exif.write()
    except:
        msg = f"Couldn't exif.write('{filename}')"
        # map_async swallows errors, so I print to the console also
        print(msg, file=sys.stderr)
        raise Exception(msg)

    return filename

def raw_convert__prototype(filename, width, height,
                           unbayer = True):

    r'''Written-in-python converter for experiments


    This is meant for testing improved processing techniques. Experiments go
    here.

    With some simple image processing along the way. This is the reference
    implementation for _raw_convert.raw_convert(). The two functions do the same
    thing, but _raw_convert.raw_convert() is written in C, with several
    optimizations in mind, so it is much faster. THIS function is good for
    testing and prototyping

    '''

    datasize_expected = width*height

    filesize = os.path.getsize(filename)

    if filesize <= datasize_expected:
        raise Exception(f"I expected {datasize_expected} bytes of pixel data, but the given file only has {filesize} bytes in it. That is too small")

    image = np.memmap(filename,
                      dtype  = np.uint8,
                      mode   = 'r',
                      offset = filesize - datasize_expected,
                      shape  = (height,width),
                      order  = 'C')

    if unbayer:
        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)

        if 1:
           lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
           clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
           lab[...,0] = clahe.apply(lab[...,0])
           image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:

            for i in range(3):

                # First, grayworld. This just means "scale the intensities to
                # move the mean to the center"
                #
                # cv2.normalize() gives me the saturated scaling with 8-bit
                # unsigned values
                x  = image[..., i]
                xr = x.ravel()
                if 0:
                    # original path
                    cv2.normalize(src       = xr,
                                  dst       = xr,
                                  # *len() to make norm_l1 = mean
                                  alpha     = 128*len(xr),
                                  norm_type = cv2.NORM_L1,
                                  dtype     = cv2.CV_8U)

                else:

                    # gray world using a better estimate of the mean (not
                    # confused by the saturations)

                    import gnuplotlib as gp
                    # gp.plot(xr, histogram=True, binwidth = 1, title = f'channel {i}', hardcopy=f"/tmp/channel{i}.png" )

                    print(f"is >= 250: {np.count_nonzero(xr >= 250)}")
                    print(f"is 0:   {np.count_nonzero(xr == 0)}")
                    i_saturated = np.logical_or((xr >= 250), (xr == 0))
                    print(f"is saturated: {np.count_nonzero(i_saturated)}")


                    print(f"minmax at start: {np.min(xr)} {np.max(xr)}")

                    q_big = 0.1
                    xmin = np.percentile(xr,       q_big)
                    xmax = np.percentile(xr, 100 - q_big)

                    xr_center = xr[ (xr >= xmin) * (xr <= xmax) ]
                    m = np.mean(xr)
                    print(f"mean(xr): {m}")
                    scale = 128. / m
                    xr = np.clip(xr.astype(float) * scale, 0, 255).astype(np.uint8)

                    print(f"minmax after grayworld: {np.min(xr)} {np.max(xr)}")

                # The input was written back to the array. The mean was scaled
                # to 128. Grayworld is done

                # I linearly stretch the intensities such that the histogram
                # covers my full range, sacrificing some stragglers at the low
                # and high ends
                if 0:
                    # original path
                    xmin = np.percentile(xr,       100./args.saturate_recip)
                    xmax = np.percentile(xr, 100 - 100./args.saturate_recip)
                elif 0:
                    # using only the non-saturated pixels for the statistics
                    xr_not_saturated = xr[~i_saturated]
                    xmin = np.percentile(xr_not_saturated,       100./args.saturate_recip)
                    xmax = np.percentile(xr_not_saturated, 100 - 100./args.saturate_recip)

                else:
                    # Smarter logic. +- 4stdev for the whole range
                    q_big = 0.1
                    xmin = np.percentile(xr,       q_big)
                    xmax = np.percentile(xr, 100 - q_big)

                    xr_center = xr[ (xr >= xmin) * (xr <= xmax) ]

                    # did gray-world, so the mean should be 128
                    m = 128 # np.mean(xr_center)
                    s = np.std( xr_center)

                    xmin = max(m - 3*s, 0)
                    xmax = min(m + 3*s, 255)


                # print(f"minmax in histogram stretch: {xmin} {xmax}")

                np.clip(xr, xmin, xmax, out=xr)
                cv2.normalize(src = xr,
                              dst = xr,
                              alpha     = 0,
                              beta      = 255,
                              norm_type = cv2.NORM_MINMAX,
                              dtype     = cv2.CV_8U)

                x[...] = xr.reshape(x.shape)

    return image

def is_raw_image(f):
    try:
        pattern = is_raw_image.pattern
    except:
        pattern = is_raw_image.pattern = \
            re.compile('.*\.(jpg|jpeg|png|tif|tiff|pgm)$', re.I)
    return not re.match(pattern, f)


if args.model is not None:
    model_intrinsics = mrcal.cameramodel(os.path.expanduser(args.model))
else:
    model_intrinsics = None

if 1:
    if args.metadata is not None:
        filenames, Rt_world_cam = \
            parse_filenames_poses_from_metadata(args.metadata,
                                                args.camera_roll_deg,
                                                args.camera_yaw_deg,
                                                args.camera_pitch_deg,
                                                args.camera_pitch_then_yaw,
                                                image_path_prefix = args.image_path_prefix,
                                                image_directory   = args.image_directory)
    else:
        filenames, Rt_world_cam = \
            parse_filenames_poses_from_exif(args.images,
                                            args.camera_roll_deg,
                                            args.camera_yaw_deg,
                                            args.camera_pitch_deg,
                                            args.camera_pitch_then_yaw,
                                            image_path_prefix = args.image_path_prefix,
                                            image_directory   = args.image_directory)

    if args.rt_world_new_world_old is not None:
        Rt_world_cam = mrcal.compose_Rt( mrcal.Rt_from_rt(np.array(args.rt_world_new_world_old)),
                                         Rt_world_cam )
    if args.rt_cam_new_cam_old is not None:
        Rt_world_cam = mrcal.compose_Rt( Rt_world_cam,
                                         mrcal.invert_Rt(mrcal.Rt_from_rt(np.array(args.rt_cam_new_cam_old))) )



images_filenames = filenames



if args.image_dims is None and any([is_raw_image(f) for f in images_filenames]):
    r0 = [f for f in images_filenames if is_raw_image(f)][0]
    raise Exception(f"Some images are raw, but --image-dims not given. First raw image: {r0}")

d0 = os.path.realpath( os.path.abspath( args.outdir ))
for f in images_filenames:
    d1 = os.path.realpath( os.path.abspath( os.path.split(f)[0] ))
    if d0 == d1:
        raise Exception("--outdir may not be a path that contains any of the input files. I want to make sure to never overwrite the input")

os.makedirs(args.outdir, mode=0o775, exist_ok = True)

latlonalt_opkdeg = compute__latlonalt_opkdeg(Rt_world_cam)
print("# filename latitude longitude altitude_m omega_deg phi_deg kappa_deg")


def output_line(f,i):
    if latlonalt_opkdeg is not None:
        print(' '.join( [f] + [str(x) for x in latlonalt_opkdeg[i]]))
    else:
        print(f"## Wrote {f}")


if args.camera_poses_only:
    for i in range(len(images_filenames)):
        if is_raw_image(images_filenames[i]):
            # Split the filename into = d + f + e (directory, basename, extension)
            d,f = os.path.split(images_filenames[i])
            f,e = os.path.splitext(f)
            output_line(f"{args.outdir}/{f}.jpg", i)
        else:
            output_line(images_filenames[i], i)
    sys.exit(0)


def process_one(i):
    filename_out = \
        process_image(images_filenames[i],
                      latlonalt_opkdeg[i] if latlonalt_opkdeg is not None else None,
                      model_intrinsics,
                      args.body_manufacturer,
                      args.body_model,
                      args.lens_model,
                      args.body_serial_number,
                      args.lens_serial_number,
                      args.serial_number,
                      args.focal_length_nominal_mm,
                      args.pixels_per_mm)

    output_line(filename_out,i)



if args.jobs <= 1:
    for i in range(len(images_filenames)):
        process_one(i)

else:

    signal_handler_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal_handler_sigint)

    pool = multiprocessing.Pool(args.jobs)
    try:
        mapresult = pool.map_async(process_one, range(len(images_filenames)))


        mapresult.get(100000000)
    except:
        pool.terminate()

    pool.close()
    pool.join()
