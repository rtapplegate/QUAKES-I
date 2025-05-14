#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps

import vnlog
import subprocess
import pyproj
import glob
import re
import os

# Defines the local cartesian coordinate system. Before use, MUST be filled-in
# by proj_context__init()
proj_context = None

def get_proj_context():
    return proj_context

def proj_context__init(latlon,
                       allow_reinit = False,
                       recenter     = True):
    r'''Initializes our locally-planar coordinate system at the given latlon

Used by ne_from_latlon() and latlon_from_ne(). This function MUST be called
before either of those

    '''

    global proj_context

    if proj_context is not None:
        if not allow_reinit:
            raise Exception("Trying to initialize an already-initialized proj_context")

    zone = 1 + int((latlon[1] + 180) // 6)
    proj_context = dict(proj = \
                        pyproj.Proj(proj           = 'utm',
                                    zone           = zone,
                                    ellps          = 'WGS84',
                                    preserve_units = False # output in meters
                                    ))

    e,n = proj_context['proj'](latlon[1], latlon[0])
    if recenter:
        proj_context['ne0'] = np.array((n,e))
    else:
        proj_context['ne0'] = (0,0)

def ne_from_latlon(latlon, get_gradients=False):
    '''Compute translated UTM (northing,easting) from a lat,lon pair

Reports (northing,easting) in meters in a common UTM coordinate system. The
local coordinate system is defined by calling proj_context__init(), which MUST
be done before calling this function

    '''

    global proj_context
    if proj_context is None:
        raise Exception("Need to initialize the context by calling proj_context__init() before computing coord transformations")

    lat = latlon[..., 0]
    lon = latlon[..., 1]
    en  = proj_context['proj'](lon,lat)
    e,n = en

    ne = nps.glue( nps.dummy(np.array(n),-1),
                   nps.dummy(np.array(e),-1),
                   axis=-1 ) - proj_context['ne0']

    if not get_gradients:
        return ne

    # gradients
    #
    # I'm using pyproj, which has some gradient support
    # (proj_context['proj'].get_factors() returns something relevant-looking).
    # But there's no clear way to convert this to a full gradient matrix, so I
    # don't even bother. I use forward differences, and call it good.
    dlatlon = 1e-6
    dne_dlatlon = np.zeros(latlon.shape + (2,), dtype=float)
    den_dlat = np.array(proj_context['proj'](lon,        lat+dlatlon)) - en
    den_dlon = np.array(proj_context['proj'](lon+dlatlon,lat        )) - en
    dne_dlatlon[...,0,0] = den_dlat[...,1] / dlatlon
    dne_dlatlon[...,1,0] = den_dlat[...,0] / dlatlon
    dne_dlatlon[...,0,1] = den_dlon[...,1] / dlatlon
    dne_dlatlon[...,1,1] = den_dlon[...,0] / dlatlon

    return ne, dne_dlatlon

def latlon_from_ne(ne):
    '''Compute lat,lon from a translated UTM (northing,easting)

Ingests a (northing,easting) in meters in a common UTM coordinate system. The
local coordinate system is defined by calling proj_context__init(), which MUST
be done before calling this function

    '''

    global proj_context
    if proj_context is None:
        raise Exception("Need to initialize the context by calling proj_context__init() before computing coord transformations")

    lon,lat = proj_context['proj'](ne[..., 1] + proj_context['ne0'][1],
                                   ne[..., 0] + proj_context['ne0'][0], inverse=True)

    return nps.glue( nps.dummy(np.array(lat),-1),
                     nps.dummy(np.array(lon),-1),
                     axis=-1 )

def massage_paths(filenames, image_path_prefix, image_directory):
    if image_path_prefix is not None:
        return [f"{image_path_prefix}/{f}" \
                for f in filenames]
    if image_directory is not None:
        return [f"{image_directory}/{os.path.basename(f)}" \
                for f in filenames]
    return filenames


def R_world_cam__from_pix4d_rpy_plane(rpy_plane,
                                      camera_roll_deg       = 0,
                                      camera_yaw_deg        = 0,
                                      camera_pitch_deg      = 0,
                                      camera_pitch_then_yaw = False):
    r'''Computes a rotation from pix4d-style euler angles

There are many interpretations of "roll,pitch,yaw". This function follows the
Pix4D conventions, defined on their support page:

  https://support.pix4d.com/hc/en-us/articles/202558969

Specifically, in the document linked on that page, and with a copy on our
server:

  https://aargh.jpl.nasa.gov/proj/uavsar/docs/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf

The "world" frame is "north,east,down"
The "body"  frame (from that pdf) is "forward,right,down"

I make an additional transformation to use mrcal's camera coordinate
system, which is (right,down,forward).

We return a rotation to map mrcal-style camera coords (right,down,forward) to
world coords (north,east,down)

    '''

    r,p,y = nps.transpose(rpy_plane)
    sy,cy = np.sin(y),np.cos(y)
    sp,cp = np.sin(p),np.cos(p)
    sr,cr = np.sin(r),np.cos(r)

    # I want to do this:
    #   Ry = np.array((( cy, -sy, 0.),
    #                  ( sy,  cy, 0.),
    #                  ( 0.,  0., 1.)))
    #
    #   Rp = np.array((( cp, 0., sp),
    #                  ( 0., 1., 0.),
    #                  (-sp, 0., cp)))
    #
    #   Rr = np.array(((1., 0.,  0.),
    #                  (0., cr, -sr),
    #                  (0., sr,  cr)))
    #
    # But each R should have shape (...,3,3), so I need to do more typing:
    Ry = np.zeros( y.shape + (3,3), dtype=float)
    Ry[...,2,2] = 1.
    Ry[...,0,0] = cy
    Ry[...,0,1] = -sy
    Ry[...,1,0] = sy
    Ry[...,1,1] = cy

    Rp = np.zeros( p.shape + (3,3), dtype=float)
    Rp[...,1,1] = 1.
    Rp[...,0,0] = cp
    Rp[...,0,2] = sp
    Rp[...,2,0] = -sp
    Rp[...,2,2] = cp

    Rr = np.zeros( r.shape + (3,3), dtype=float)
    Rr[...,0,0] = 1.
    Rr[...,1,1] = cr
    Rr[...,1,2] = -sr
    Rr[...,2,1] = sr
    Rr[...,2,2] = cr

    R_world_body = nps.matmult(Ry,Rp,Rr)

    # The rpy we just processed is from a PLANE. We need an extra transformation
    # to remap this into CAMERA coords. We have another set of (yaw,pitch,roll),
    # describing the pose of the camera in respect to the plane. In my one test
    # case so far I have the cameras looking ~90deg to the left and are mounted
    # sideways (roll ~ 90deg), and are looking some amount down (non-zero
    # pitch). I need to incorporate the rotations AND take into account the
    # different coord systems. The plane (body) coordinate system is
    # (forward,right,down), while the mrcal camera coord system is
    # (right,down,forward). Testing in my one case so far tells me that for sp =
    # sin(pitch), cp = sin(pitch) I should get
    #
    #   R_cam_body = np.array((
    #       # cam-right in plane coords
    #       (0,  sp, cp),
    #       # cam-down in plane coords
    #       (-1., 0., 0.),
    #       # cam-forward in plane coords
    #       (0., -cp, sp)))
    #
    # I engineer a sequence of Euler angle transformations that produces this
    # result, and I validate it using this script:
    #
    #    import numpy as np
    #    import numpysane as nps
    #    import sympy
    #
    #    sp,cp = sympy.symbols(('sp','cp'))
    #    sy,cy = sympy.symbols(('sy','cy'))
    #    sr,cr = sympy.symbols(('sr','cr'))
    #
    #    # yaw and roll = 90deg (camera is sideways, looking out to the side)
    #    sy = 1
    #    cy = 0
    #
    #    sr = 1
    #    cr = 0
    #
    #    R_cam_body_reorder = np.array(((0,1,0),
    #                                   (0,0,1),
    #                                   (1,0,0)))
    #
    #    Ryaw   = np.array((( cy, 0, sy),
    #                       (  0, 1,  0),
    #                       (-sy, 0, cy)))
    #    Rpitch = np.array(((1,  0,  0),
    #                       (0, cp,-sp),
    #                       (0, sp, cp)))
    #    Rroll  = np.array((( cr, sr, 0),
    #                       (-sr, cr, 0),
    #                       (  0,  0, 1)))
    #
    #    R_cam_body = nps.matmult(Rroll, Rpitch, Ryaw, R_cam_body_reorder)
    #
    #    print(repr(R_cam_body))
    #
    # It reports
    #
    #   array([[0, sp, cp],
    #          [-1, 0, 0],
    #          [0, -cp, sp]], dtype=object)
    #
    # Which matches the desired R_cam_body above. Without hardcoding the
    # yaw,roll: and I to get:
    sr = np.sin(camera_roll_deg  * np.pi/180.0)
    cr = np.cos(camera_roll_deg  * np.pi/180.0)
    sp = np.sin(camera_pitch_deg * np.pi/180.0)
    cp = np.cos(camera_pitch_deg * np.pi/180.0)
    sy = np.sin(camera_yaw_deg   * np.pi/180.0)
    cy = np.cos(camera_yaw_deg   * np.pi/180.0)

    # camera (right,down,forward) <- plane (forward,right,down)
    R_cam_body_reorder = np.array(((0,1,0),
                                   (0,0,1),
                                   (1,0,0)))

    # Initially, the camera is oriented horizontally, looking ahead, along the
    # main axis of the plane.
    #
    # We yaw around the vertical axis (camera y); assuming no
    # --camera-pitch-then-yaw
    Ryaw   = np.array((( cy, 0, sy),
                       (  0, 1,  0),
                       (-sy, 0, cy)))

    # Then we pitch around the camera's new horizontal axis (camera x); assuming
    # no --camera-pitch-then-yaw
    Rpitch = np.array(((1,  0,  0),
                       (0, cp,-sp),
                       (0, sp, cp)))

    # Finally we rotate the camera along its view axis (camera z)
    Rroll  = np.array((( cr, sr, 0),
                       (-sr, cr, 0),
                       (  0,  0, 1)))

    if camera_pitch_then_yaw:
        R_cam_body = nps.matmult(Rroll, Ryaw, Rpitch, R_cam_body_reorder)
    else:
        R_cam_body = nps.matmult(Rroll, Rpitch, Ryaw, R_cam_body_reorder)

    return nps.matmult(R_world_body, nps.transpose(R_cam_body))

def Rt_world_cam__from_pix4d_pose_plane(rpydeg_latlonalt,
                                        camera_roll_deg       = 0,
                                        camera_yaw_deg        = 0,
                                        camera_pitch_deg      = 0,
                                        camera_pitch_then_yaw = False):

    r'''Computes a transformation from EXIF metadata

Returns a transformation (in an (Ncameras,4,3) array) to map mrcal-style camera
coords (right,down,forward) to world coords (north,east,down).

    '''

    rpy = rpydeg_latlonalt[:,:3] * np.pi/180.

    # shape (...,3,3)
    R_world_cam = R_world_cam__from_pix4d_rpy_plane(rpy,
                                                    camera_roll_deg       = camera_roll_deg,
                                                    camera_yaw_deg        = camera_yaw_deg,
                                                    camera_pitch_deg      = camera_pitch_deg,
                                                    camera_pitch_then_yaw = camera_pitch_then_yaw)

    lat,lon,alt = nps.transpose(rpydeg_latlonalt[:, 3:6])

    # shape (...,1,3)
    ned = nps.glue( # shape (...,1,2)
                    ne_from_latlon(# shape (...,1,2)
                                   nps.glue(# shape (...,1,1)
                                       nps.mv( lat,-1,-3),
                                       nps.mv( lon,-1,-3),
                                       axis=-1)),
                    # shape (...,1,1)
                    nps.mv(-alt,-1,-3),
                    axis = -1 )

    Rt_world_cam = nps.glue(R_world_cam, ned,
                            axis = -2)
    return Rt_world_cam

def parse_filenames_poses_from_metadata(metadata_filename,
                                        camera_roll_deg,
                                        camera_yaw_deg,
                                        camera_pitch_deg,
                                        camera_pitch_then_yaw,
                                        *,
                                        image_path_prefix = None,
                                        image_directory   = None):
    r'''Parse metadata for the image filenames and the world-camera poses

This function tries to be very flexible. The fields it knows about:

  - latitude, longitude:              Degrees. Position of the plane (and camera)
  - altitude_ft or altitude_m:        altitude of the plane (and camera)
  - roll_deg, pitch_deg, heading_deg: Orientation of the AIRPLANE
  - omega_deg, phi_deg, kappa_deg:    Orientation of the CAMERA.
  - filename:                         the image captured by a camera with this pose

    '''

    # I'd like to use np.loadtxt(), but it will fail on any non-numerical data.
    # So I parse the filename separately
    with open(metadata_filename, 'r') as f:
        filenames = \
            subprocess.check_output(('vnl-filter',
                                     '--skipcomments',
                                     '-p',
                                     'filename',),
                                    shell    = False,
                                    encoding = 'ascii',
                                    stdin    = f). \
                      splitlines()[1:] # [1:] to ignore vnl header

    with open(metadata_filename, 'r') as f:
        read_text_process = \
            subprocess.Popen(('vnl-filter',
                              '--skipcomments',
                              '-p',
                              '!filename',),
                             shell    = False,
                             encoding = 'ascii',
                             stdin    = f,
                             stdout   = subprocess.PIPE)

        metadata,list_keys,dict_key_index = vnlog.slurp(read_text_process.stdout)

        # output should be exhausted already, but I finish it just in case...
        read_text_process.communicate()


    if len(filenames) != len(metadata):
        print(f"Parsing of {metadata_filename} failed: read {len(filenames)} image filenames, but only {len(metadata)} rows of metadata",
              file=sys.stderr)
        sys.exit(1)


    dict_key_index_lowercase = dict()
    for k in dict_key_index:
        dict_key_index_lowercase[k.lower()] = dict_key_index[k]

    def get_column_from_any(*keys):
        for k in keys:
            if k in dict_key_index:
                return metadata[:,dict_key_index_lowercase[k]], k
        print(f"Parsing of {metadata_filename} failed: must contain any of {keys}",
              file=sys.stderr)
        sys.exit(1)

    N = len(filenames)

    # Might be world-airplane or world-camera at this point
    rpydeg_latlonalt = np.zeros((N,6), dtype=float)

    # First we parse out the position. This must be present in all cases
    rpydeg_latlonalt[:,3],_ = get_column_from_any('lat',
                                                  'latitude')
    rpydeg_latlonalt[:,4],_ = get_column_from_any('lon',
                                                  'long',
                                                  'longitude')
    rpydeg_latlonalt[:,5],key = get_column_from_any('altitude_ft','alt_ft',
                                                    'altitude_m', 'alt_m')
    if re.search("_ft$", key, re.I):
        # ft -> m
        rpydeg_latlonalt[:,5] *= 12.*2.54/100.

    proj_context__init( # lat,lon
        (rpydeg_latlonalt[0,3], rpydeg_latlonalt[0,4] ))

    # Now we parse out the orientation. This will be different, depending on
    # whether we have a world-airplane transform or a world-camera transform
    if 'roll_deg' in dict_key_index_lowercase:
        # We must have a world-airplane transform
        rpydeg_latlonalt__airplane        = rpydeg_latlonalt
        rpydeg_latlonalt__airplane[:,0],_ = get_column_from_any('roll_deg')
        rpydeg_latlonalt__airplane[:,1],_ = get_column_from_any('pitch_deg')
        rpydeg_latlonalt__airplane[:,2],_ = get_column_from_any('heading_deg',
                                                                'yaw_deg')

        # shape (Ncam, 4,3)
        Rt_world_cam = \
            Rt_world_cam__from_pix4d_pose_plane(rpydeg_latlonalt__airplane,
                                                camera_roll_deg       = camera_roll_deg,
                                                camera_yaw_deg        = camera_yaw_deg,
                                                camera_pitch_deg      = camera_pitch_deg,
                                                camera_pitch_then_yaw = camera_pitch_then_yaw)

    else:
        # We must have a world-camera transform

        # We have a cam-world transform already encoded in the given
        # omega,phi,kappa. The meaning of these is defined on their support
        # page:
        #
        #   https://support.pix4d.com/hc/en-us/articles/202558969
        #
        # Specifically, in the document linked on that page, and with a copy on our
        # server:
        #
        #   https://aargh.jpl.nasa.gov/proj/uavsar/docs/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf

        opkdeg_latlonalt        = rpydeg_latlonalt
        opkdeg_latlonalt[:,0],_ = get_column_from_any('omega_deg')
        opkdeg_latlonalt[:,1],_ = get_column_from_any('phi_deg')
        opkdeg_latlonalt[:,2],_ = get_column_from_any('kappa_deg')

        o,p,k = nps.transpose(opkdeg_latlonalt[...,:3]) * np.pi/180.
        so,co = np.sin(o),np.cos(o)
        sp,cp = np.sin(p),np.cos(p)
        sk,ck = np.sin(k),np.cos(k)


        Ro = np.zeros( o.shape + (3,3), dtype=float)
        Ro[...,0,0] = 1.
        Ro[...,1,1] = co
        Ro[...,1,2] = -so
        Ro[...,2,1] = so
        Ro[...,2,2] = co

        Rp = np.zeros( p.shape + (3,3), dtype=float)
        Rp[...,1,1] = 1.
        Rp[...,0,0] = cp
        Rp[...,0,2] = sp
        Rp[...,2,0] = -sp
        Rp[...,2,2] = cp

        Rk = np.zeros( k.shape + (3,3), dtype=float)
        Rk[...,2,2] = 1.
        Rk[...,0,0] = ck
        Rk[...,0,1] = -sk
        Rk[...,1,0] = sk
        Rk[...,1,1] = ck

        R_EB = nps.matmult(Ro,Rp,Rk)

        # The pdf document above describes a transformation C_EB.
        #
        # I'm assuming the Earth is locally flat, so the coord system E is always
        # aligned with the ground. Docs say this coord system is (E,N,up)
        R_world_E = np.array(((0,1, 0),
                              (1,0, 0),
                              (0,0,-1)))

        # The coord system B is aligned with the camera. It is (right,up,back)
        R_B_cam = np.array(((1, 0, 0),
                            (0,-1, 0),
                            (0, 0,-1)))

        R_world_cam = nps.matmult(R_world_E, R_EB, R_B_cam)

        t_world_cam = nps.glue( ne_from_latlon(opkdeg_latlonalt[...,3:5]),
                                -opkdeg_latlonalt[...,(5,)],
                                axis = -1)
        Rt_world_cam = nps.glue(R_world_cam,
                                nps.dummy(t_world_cam, -2),
                                axis = -2)

    return \
        massage_paths(filenames, image_path_prefix, image_directory), \
        Rt_world_cam

def get__filenames__rpydeg_latlonalt__from_anafi_parrot(g):
    r'''Parse EXIF poses from Anafi parrot images

The returned poses are of the CAMERA'''


    # Required to be able to touch Xmp.Camera tags
    try:
        pyexiv2.register_namespace('https://support.pix4d.com/hc/en-us/articles/205732309-EXIF-and-XMP-tag-information-read-by-Pix4D-Desktop/', 'Camera')
    except:
        pass

    filenames = sorted(glob.glob(os.path.expanduser(g)))
    if len(filenames) == 0:
        raise Exception(f"No files matched the glob '{g}'")

    filenames = np.array(filenames)

    @nps.broadcast_define( ((),), (6,), out_kwarg='out' )
    def get_rpydeg_latlonalt(f, out):
        meta = pyexiv2.ImageMetadata(f)
        meta.read()

        try:
            roll  = float(meta['Xmp.drone-parrot.CameraRollDegree' ].value)
            pitch = float(meta['Xmp.drone-parrot.CameraPitchDegree'].value)
            yaw   = float(meta['Xmp.drone-parrot.CameraYawDegree'  ].value)
        except:
            raise Exception("Could not read anafi-parrot EXIF tags. If not using an anafi parrot, you MUST pass exactly one of (--ins-vnl-plane --latlonalt-opkdeg-vnl)")

        def accum_dms(n):
            v = meta[n].value
            x = 0
            s   = 1.
            for dms in v:
                x += float(dms)/s
                s *= 60
            return x

        lat = accum_dms('Exif.GPSInfo.GPSLatitude')
        lon = accum_dms('Exif.GPSInfo.GPSLongitude')
        if meta['Exif.GPSInfo.GPSLatitudeRef' ].value == 'S': lat *= -1
        if meta['Exif.GPSInfo.GPSLongitudeRef'].value == 'W': lon *= -1

        alt = float(meta['Exif.GPSInfo.GPSAltitude'].value)
        if meta['Exif.GPSInfo.GPSAltitudeRef'].value != '0':
            raise Exception("Only Exif.GPSInfo.GPSAltitudeRef == 0 is supported: above sea level")

        out[0] = roll
        out[1] = pitch
        out[2] = yaw
        out[3] = lat
        out[4] = lon
        out[5] = alt



    N = len(filenames)
    rpydeg_latlonalt = np.zeros((N, 6), dtype=float)
    get_rpydeg_latlonalt(filenames, out=rpydeg_latlonalt)

    return filenames,rpydeg_latlonalt

def parse_filenames_poses_from_exif(images,
                                    camera_roll_deg,
                                    camera_yaw_deg,
                                    camera_pitch_deg,
                                    camera_pitch_then_yaw,
                                    *,
                                    image_path_prefix = None,
                                    image_directory   = None):

    raise("""This path is currently broken. Previously this only ever worked for
    the anafi parrot. There's an old have_plane_poses argument that doesn't
    exist anymore. Bring this back, and make it work for pix4d images too""")


    # We're reading EXIF tags from an Anafi Parrot
    filenames, rpydeg_latlonalt = \
        get__filenames__rpydeg_latlonalt__from_anafi_parrot(images)

    proj_context__init( # lat,lon
        (rpydeg_latlonalt__plane[0,3], rpydeg_latlonalt__plane[0,4] ))

    # shape (Ncam, 4,3)
    Rt_world_cam = \
        Rt_world_cam__from_pix4d_pose_plane(rpydeg_latlonalt,
                                            have_plane_poses = False,
                                            camera_roll_deg  = camera_roll_deg,
                                            camera_yaw_deg   = camera_yaw_deg,
                                            camera_pitch_deg = camera_pitch_deg,
                                            camera_pitch_then_yaw = camera_pitch_then_yaw)

    return \
        massage_paths(filenames, image_path_prefix, image_directory), \
        Rt_world_cam
