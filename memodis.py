import os, sys, gc
import numpy as np


from collections import defaultdict
#from matplotlib.pyplot import imread, imsave
from scipy.misc import imread, imsave

PROG_NAME = 'memodis'
VERSION = '1.0.0'
DESCRIPTION = 'A simple cellular membrane molecule diffusion simulator.'

DFLT_N_PARTICLES = 1000
DFLT_N_SAMPLES = 100
DFLT_N_STEPS = 10000
DFLT_TIMESTEP = 0.05
DFLT_D_BOUND = 0.01
DFLT_D_FREE = 25.0
DFLT_INIT_STEPS = 1000
DFLT_K_ON = 0.2
DFLT_ALT_K_ON = 0.2 
DFLT_K_OFF = 0.1
DFLT_ALT_K_OFF = 0.15
DFLT_MAX_OCC = 1

GRAD_STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRAD_RED   = [0.0, 0.2,  0.5, 0.0,  1.0]
GRAD_GREEN = [0.0, 0.2,  0.0, 1.0,  1.0]
GRAD_BLUE  = [0.0, 0.2,  0.5, 0.0,  0.0]  

VALID_ZONES = ('origin','adjust','roi')
IMG_FILTER = [[0.5, 1.0, 0.5],
              [1.0, 1.0, 1.0],
              [0.5, 1.0, 0.5]]
# Input is 
# - an image
# - zones file
#   + origin x, y, radius
#   + adjust x, y, radius
#   + roi x, y, radius, group

# Output is
# - an image
# - and particle coords
# - a counts table
# - movie?

NEWLINE_CHARS = 0


def report(msg, line_return):
 
  global NEWLINE_CHARS

  if line_return:
    fmt = '\r%%-%ds' % max(NEWLINE_CHARS, len(msg))
    sys.stdout.write(fmt % msg)
    sys.stdout.flush()
    NEWLINE_CHARS = len(msg)
    
  else: 
    if NEWLINE_CHARS:
      print('')
      
    print(msg)
    NEWLINE_CHARS = 0


def warn(msg, prefix='WARNING', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)

 
def critical(msg, prefix='FAILURE', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)
  sys.exit(0)


def info(msg, prefix='INFO', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)

  
def check_invalid_file(file_path, critical=True):
  
  msg = ''
  
  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist' % file_path
 
  elif not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file' % file_path
  
  elif os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size ' % file_path
    
  elif not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable' % file_path

  if msg and critical:
    critical(msg)
  
  return msg


def test_imports():
  import sys
  from distutils.core import run_setup
  
  critical = False

  try:
    from scipy import ndimage  
    
  except ImportError as err:
    critical = True
    warn('Critical Python module "scipy" is not installed or accessible')
  
  try:
    import numpy
    
  except ImportError as err:
    critical = True
    warn('Critical Python module "numpy" is not installed or accessible')

  try:
    import cython
  except ImportError as err:
    critical = True
    warn('Critical Python module "cython" is not installed or accessible')
  
  try:
    import dyn_sim
  
  except ImportError as err:
    warn('Utility C/Cython code not compiled. Attempting to compile now...')    
    run_setup('setup_cython.py', ['build_ext', '--inplace'])
    
    try:
      import dyn_sim
      warn('MEMODIS C/Cython code compiled. Please re-run command.')
      sys.exit(0)
      
    except ImportError as err:
      critical = True
      warn('Utility C/Cython code compilation/import failed')   
    
  if critical:
    critical('MEMODIS cannot proceed because critical Python modules are not available')
    
    
def _read_zone_file(zone_path):
  
  origins = []
  adjusts = []
  rois = []
  
  with open(zone_path) as file_obj:
    for line in file_obj:
      if line[0] == '#':
        continue
      
      line = line.strip()
      data = line.split()
      
      if not data:
        continue
        
      key = data[0].lower()
      
      if key not in VALID_ZONES:
        critical('Zone type in file %s must be "origin", "adjust" or "roi" not %s'% (zone_path, key))
      
      if key == 'origin':
        if len(data) != 4:
          critical('Zone origin lines must have 4 witespace-separate columns: "origin", x_pos, y_pos, radius')
        
        try:
          vals = [float(x) for x in data[1:4]]        
        except ValueError:
          critical('Zone origin line "%s" values not interpretable' % line)
        
        origins.append(vals)
        
      elif key == 'adjust':
        if len(data) != 4:
          critical('Zone ajustment lines must have 4 witespace-separate columns: "adjust", x_pos, y_pos, radius ')
        
        try:
          vals = [float(x) for x in data[1:4]]        
        except ValueError:
          critical('Zone ajustment line "%s" values not interpretable' % line)
        
        adjusts.append(vals)
        
      elif key == 'roi':
        if len(data) != 5:
          critical('Zone region of interest lines must have 5 witespace-separate columns: "roi", x_pos, y_pos, radius, group')
        
        try:
          vals = [float(x) for x in data[1:4]] + [data[4]]
                  
        except ValueError:
          critical('Zone region of interest line "%s" values not interpretable' % line)
        
        rois.append(vals)
        
  if not origins:
    critical('Zone file must specify at least one particle origin')

  if not adjusts:
    warn('Zone file does not specifiy any (un)bining adjustment zomes')
    
  if not rois:
    warn('Zone file does not specify any regions of interest.')
  
  return origins, adjusts, rois


def run_dynamics(bg_pixmap, coords, out_img_dir, monitor_regions, adjust_zones,
                 n_particles, n_samples, n_steps, timestep, d_bound, d_free,
                 init_steps, k_on, alt_k_on, k_off, alt_k_off, max_occ,
                 pixel_decay=1, pixel_width=2, free_color=(255, 0, 0), bound_color=(255, 255, 0)):
  
  from dyn_sim import dynamics_steps, add_pixmap_particles

  pixmap = np.array(bg_pixmap)
  n_coords = len(coords)
  
  print pixmap.max(axis=(0,1))
  
  nz = bg_pixmap[:,:,:3].sum(axis=2) > 0
  
  
  bg_pixmap[nz] = (64, 64, 64, 64)
  
  # Adapt parameters
  # - units of seconds converted to timestep probabilities etc.
  # - diffucion represented by Gaussian sigma
  
  sigma_free = (d_free * timestep * 2.0) ** 0.5
  sigma_bound = (d_bound * timestep * 2.0) ** 0.5
  
  p_bind = k_on * timestep
  p_unbind = k_off * timestep
  
  alt_p_bind = alt_k_on * timestep
  alt_p_unbind = alt_k_off * timestep

  # monitor_regions [(x_pos, y_pos, radius),]  
  
  # Different particles could potentially be modelled differently
  particle_params = np.array([[sigma_free, sigma_bound, p_bind, p_unbind, alt_p_bind, alt_p_unbind]] * n_coords) 

  info('  .. initial equilibriation')
  coords, region_counts = dynamics_steps(bg_pixmap, coords, particle_params, adjust_zones,
                                         monitor_regions, init_steps, max_occ)
  
  count_data = []
  
  n, m, dims = pixmap.shape
  density = np.zeros((n,m), float)
  colors = np.array(free_color+bound_color, np.uint8)
  
  j = 0
  for i in range(n_samples):
    local_steps = max(1, n_steps/n_samples)
 
    # region_counts # num_regions, 2 (free/bound)
    coords, region_counts = dynamics_steps(bg_pixmap, coords, particle_params, adjust_zones,
                                           monitor_regions, local_steps, max_occ)
        
    counts = region_counts.astype(float) / float(local_steps) # average totals per step
    bound = counts[:,0]
    free = counts[:,1]

    count_data.append((bound, free))
                      
    #np.save('npy/particle_coords_%s_%03d.npy' % (label, i), coords)
    
    if out_img_dir:
      add_pixmap_particles(pixmap, bg_pixmap, coords, colors, pixel_decay, pixel_width)
      file_path = os.path.join(out_img_dir, 'step_%05d.png')
      imsave(file_path % i, pixmap)
      
    pixel_points = coords.astype(int)
    hist, ex, ey = np.histogram2d(pixel_points[:,0], pixel_points[:,1], range=((0,n), (0,m)), bins=(n,m))
    density += hist
    
    if i % 10 == 0:
      gc.collect()
    
    j += local_steps
    info("  .. done %d of %d" % (j, n_steps), line_return=True)
  
  count_data = np.array(count_data).T

  # Normalise and clip extrena
  density -= density.min()
  density /= density.max()
  
  density = np.clip(density, 0.01, 0.99)
  density -= density.min()
  density /= density.max()  
  
  return density, coords, count_data
  


def mem_mol_diff_sim(image_path, zone_path, out_dir, n_particles=DFLT_N_PARTICLES, n_samples=DFLT_N_SAMPLES,
                     n_steps=DFLT_N_STEPS, timestep=DFLT_TIMESTEP, d_bound=DFLT_D_BOUND, d_free=DFLT_D_FREE,
                     init_steps=DFLT_INIT_STEPS, k_on=DFLT_K_ON, alt_k_on=DFLT_ALT_K_ON,
                     k_off=DFLT_K_OFF, alt_k_off=DFLT_ALT_K_OFF, max_occ=DFLT_MAX_OCC):
  
  from scipy import ndimage  
  from numpy.random import uniform
  
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    
  check_invalid_file(image_path)
  check_invalid_file(zone_path)
  
  origins, adjusts, rois = _read_zone_file(zone_path)
  
  out_count_file = os.path.join(out_dir, 'particle_counts.tsv')
  out_ratio_file = os.path.join(out_dir, 'region_bound_ratios.tsv')
  out_img_dir = os.path.join(out_dir, 'sample_images')
  out_img_path = os.path.join(out_dir, 'particle_density.png')
  
  if not os.path.exists(out_img_dir):
    os.mkdir(out_img_dir)
  
  try:
    #bg_pixmap = (imread(image_path) * 255).astype(np.uint8)
    bg_pixmap = imread(image_path)
  
  except Exception as err:
    report(str(err), False)
    critical('Failed to load image file.')  
  
  info('Running memodis on membrane image %s' % image_path)
  info('Using zones %s' % zone_path)
  info('Simulating %d particles per origin for %d steps' % (n_particles, n_steps))
  
  # From origins
  start_coords = np.empty((len(origins)*n_particles, 2), float)
  h, w = bg_pixmap.shape[:2]
  i = 0
  for x0, y0, r in origins:
    deltas = uniform(-r, r, (n_particles, 2))    
    d2 = (deltas * deltas).sum(axis=1)    
    invalid = (d2 > r*r).nonzero()[0]
    
    while len(invalid):
      deltas[invalid] = uniform(-r, r, (len(invalid), 2))
      d2 = (deltas * deltas).sum(axis=1)
      invalid = (d2 > r*r).nonzero()[0]
      
    origin_coords = deltas + np.array([y0, x0])
    
    # Wrap around edges
    origin_coords[:,0] = origin_coords[:,0] % h
    origin_coords[:,1] = origin_coords[:,1] % w
    
    start_coords[i:i+n_particles] = origin_coords
    i += n_particles
  
  roi_groups = defaultdict(list)
  monitor_regions = []
  for i, (x, y, r, g) in enumerate(rois):
    roi_groups[g].append(i)
    monitor_regions.append([x,y,r])
  
  monitor_regions = np.array(monitor_regions, float)
    
  adjust_zones = np.array(adjusts)
  
    
  density, coords, count_data = run_dynamics(bg_pixmap, start_coords, out_img_dir,
                                             monitor_regions, adjust_zones,
                                             n_particles, n_samples, n_steps, timestep,
                                             d_bound, d_free, init_steps,
                                             k_on, alt_k_on, k_off, alt_k_off, max_occ)
    
  red = np.interp(density, GRAD_STOPS, GRAD_RED)
  green = np.interp(density, GRAD_STOPS, GRAD_GREEN)
  blue = np.interp(density, GRAD_STOPS, GRAD_BLUE)
  
  red = ndimage.convolve(red, IMG_FILTER)
  green = ndimage.convolve(green, IMG_FILTER)
  blue = ndimage.convolve(blue, IMG_FILTER)  
  
  color_pixmap = np.array([red, green, blue]).T
  color_pixmap -= color_pixmap.min()
  color_pixmap /= color_pixmap.max()
  
  color_pixmap = np.clip(color_pixmap, 0.0, 1.0)
  color_pixmap = (color_pixmap * 255).astype(np.uint8)   
   
  imsave(out_img_path, color_pixmap)
  info('Written %s' % out_img_path)
  
  
  # Write count data
  #  - count_data isfor each zone, (bound, free)
  
  with open(out_count_file, 'w') as out_file_obj:
    write = out_file_obj.write

     
    for j, region in enumerate(monitor_regions):
      x_pos, y_pos, radius = monitor_regions[j]
      line = '## region:%d x_pos:%d y_pos:%d radius;%d\n' % (j, x_pos, y_pos, radius)
      write('\t'.join(line) + '\n')
       
      head = ['#region', 'step', 'num_bound', 'num_free']
      write('\t'.join(head) + '\n')
    
      for i, (bound, free) in enumerate(count_data): # for each sample
        line = '%d\t%d\t%d\t%d\n'
        write(line % (j, i, bound[j], free[j]))

  
  # Write ratios
  
  # Average over all samples
  mean_count_data = count_data.mean(axis=0)
  
  with open(out_ratio_file, 'w') as out_file_obj:
    write = out_file_obj.write

    head = ['#group_a', 'group_b', 'regions_a', 'regions_b', 'bound_a', 'bound_b', 'mean_a', 'mean_b', 'ratio_a/b']
    write('\t'.join(head) + '\n')
    groups = sorted(roi_groups)
    
    if len(groups) > 1:
      bound_dict = defaultdict(list)
      free_dict = defaultdict(list)
    
      for g in groups:
        for j in roi_groups[g]:
          bound = mean_count_data[0,j]
          free = mean_count_data[1,j]
          bound_dict[g].append(bound)
          free_dict[g].append(free)
    
      for i, a in enumerate(groups[:-1]):
        regions_a = len(bound_dict[a])
        bound_a = sum(bound_dict[a])
        mean_a = np.mean(bound_dict[a])
        
        for b in groups[i:]:
          regions_b = len(bound_dict[a])
          bound_b = sum(bound_dict[a])
          mean_b = np.mean(bound_dict[a])
          
          if mean_b:
            ratio = '%.4f' % (mean_a/mean_b)
          else:
            ratio = ''
          
          data = (a, b, regions_a, regions_b, bound_a, bound_b, mean_a, mean_b, ratio)
          line = '%s\t%s\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%s\n' % data
          write(line)

test_imports()


def main(argv=None):

  from argparse import ArgumentParser
  
  if argv is None:
    argv = sys.argv[1:]
  
  # 'Example use: ./memodis example/cell_boundaries.png example/zones_example.tsv -o example/ '
  
  epilog = 'For further help email tjs23@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='IMAGE_FILE',dest='i',
                         help='Cell membrane boundary pixmap image in PNG format (other formats may be usable if the PIL library is available).' \
                              ' Non-membrane/cytompasm areas must be perfectly black. Membrane pixels may be any other colour.')
  
  arg_parse.add_argument(metavar='ZONE_FILE', dest='z',
                         help='A white-space separated text file specifying circular zones for particle origins, (un)bining' \
                              ' adjustments and monitored regions of interest. See the main GitHub README for a description of the file format.')

  arg_parse.add_argument('-o', metavar='OUT_DIR', dest='o', default='.',
                         help='Optional output directory for writing image and count data. Defaults to the current working directory.')
  
  arg_parse.add_argument('-p', '--num-particles', default=DFLT_N_PARTICLES, metavar='NUM_PARTICLES', type=int, dest='p',
                         help='The number of simulated particles for EACH origin zone, as specified in the zone file. Default: %d' % DFLT_N_PARTICLES)

  arg_parse.add_argument('-s', '--num_samples', default=DFLT_N_SAMPLES, metavar='NUM_SAMPLES', type=int, dest='s',
                         help='The number of points in the simulation to record particle positions,' \
                              ' after the initialization period. Default: %d' % DFLT_N_SAMPLES)

  arg_parse.add_argument('-n', '--num_steps', default=DFLT_N_STEPS, metavar='NUM_SIM_STEPS', type=int, dest='n',
                         help='Number of discrete simulation diffusion steps after the initialization period. Default: %d' % DFLT_N_STEPS)

  arg_parse.add_argument('-t', '--timestep', default=DFLT_TIMESTEP, metavar='TIMESTEP', type=float, dest='t',
                         help='The time interval corresponding to each simulation step. Default: %.4f' % DFLT_TIMESTEP)
  
  arg_parse.add_argument('-b', '--diff-bound', default=DFLT_D_BOUND, metavar='DIFF_BOUND', type=float, dest='b',
                         help='Diffusion coefficient for membrane bound particles, in units of um^2/s. Default: %.4f' % DFLT_D_BOUND)
  
  arg_parse.add_argument('-f', '--diff-free', default=DFLT_D_FREE, metavar='DIFF_FREE', type=float, dest='f',
                         help='Diffusion coefficient for free, unbound particles, in units of um^2/s. Default: %.4f' % DFLT_D_FREE)

  arg_parse.add_argument('-is', '--num-init-steps', default=DFLT_INIT_STEPS, metavar='NUM_INIT_STEPS', type=int, dest='is',
                         help='Number of initialization simulation steps (to equilibriate the system)' \
                              ' before recording particle postions. Default: %d' % DFLT_INIT_STEPS)

  arg_parse.add_argument('-a', '--k-on', default=DFLT_K_ON, metavar='ASSOC_CONST', type=float, dest='a',
                         help='The general membrane association constant, K_on. Default: %.4f' % DFLT_K_ON)
  
  arg_parse.add_argument('-a2', '--alt-k-on', default=DFLT_ALT_K_ON, metavar='ALT_ASSOC_CONST', type=float, dest='a2',
                         help='The special membrane association constant, K_on, for adjusted zones. Default: %.4f' % DFLT_ALT_K_ON)
  
  arg_parse.add_argument('-d', '--k-off', default=DFLT_K_OFF, metavar='DISSOC_CONST', type=float, dest='d',
                         help='The general membrane dissociation constant, K_off. Default: %.4f' % DFLT_K_OFF)

  arg_parse.add_argument('-d2', '--alt-k-off', default=DFLT_ALT_K_OFF, metavar='ALT_DISSOC_CONST', type=float, dest='d2',
                         help='The special membrane dissociation constant, K_off, for adjusted zones. Default: %.4f' % DFLT_ALT_K_OFF)

  arg_parse.add_argument('-m', '--max-occ', default=DFLT_MAX_OCC, metavar='MAX_OCCUPANCY', type=int, dest='m',
                         help='Maximum per-pixel particle occupancy at membrane. Default: %d' % DFLT_MAX_OCC)
  
  args = vars(arg_parse.parse_args(argv))

  image_path = args['i']
  zone_path = args['z']
  out_dir = args['o']
  n_particles = args['p']
  n_samples = args['s']
  n_steps = args['n']
  timestep = args['t']
  d_bound = args['b']
  d_free = args['f']
  init_steps = args['is']
  k_on = args['a']
  alt_k_on = args['a2']
  k_off = args['d']
  alt_k_off = args['d2']
  max_occ = args['m']
                   
  mem_mol_diff_sim(image_path, zone_path, out_dir, n_particles, n_samples, n_steps, timestep,
                   d_bound, d_free, init_steps, k_on, alt_k_on, k_off, alt_k_off, max_occ)
  
if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

  
